import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from datetime import datetime 
import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import movie_reviews

reviews = pd.read_csv('gpAllReviewsCash.csv')
reviews = reviews.drop(columns=['replyDate','id','replyText','title','url','userImage','userName'])
grouped = reviews.groupby('date')


## Getting the dates and sorting them
dates = []
for x in grouped.groups.keys():
    dates.append(x)
dates.sort(key = lambda date: datetime.strptime(date, '%B %d, %Y'))


## Number of total ratings per day
count = []
for x in grouped.groups.keys():
    count.append(len(grouped.get_group(x)))


## Getting number of n-star reviews, plus some extra metrics
col1 = []
col2 = []
col3 = []
col4 = []
col5 = []
for day, group in grouped:
    one = 0
    two = 0
    thr = 0 
    four = 0
    five = 0
    for row in group.itertuples():
        if (row.score == 1):
            one = one+1
        elif (row.score == 2):
            two = two+1
        elif (row.score == 3):
            thr = thr+1
        elif (row.score == 4):
            four = four+1
        elif (row.score == 5):
            five = five+1
    col1.append(one)
    col2.append(two)
    col3.append(thr)
    col4.append(four)
    col5.append(five)
zippedList =  list(zip(col1, col2, col3, col4, col5))
scores = pd.DataFrame(zippedList,columns=['1','2','3','4','5'],index=dates)
scores['total'] = scores['1'] +scores['2']+scores['3']+scores['4']+scores['5']
scores['%1'] = scores['1']/scores['total']
scores['%2'] = scores['2']/scores['total']
scores['%3'] = scores['3']/scores['total']
scores['%4'] = scores['4']/scores['total']
scores['%5'] = scores['5']/scores['total']


fig = go.Figure()
fig.add_trace(go.Bar(x=scores.index,
                y=scores['1'],
                name='1 star',
                marker_color='red',
                hovertext = round(scores['%1'],3)
                ))
fig.add_trace(go.Bar(x=scores.index,
                y=scores['2'],
                name='2 star',
                marker_color='orange',
                hovertext = round(scores['%2'],3)
                ))
fig.add_trace(go.Bar(x=scores.index,
                y=scores['3'],
                name='3 star',
                marker_color='grey',
                hovertext = round(scores['%3'],3)
                ))
fig.add_trace(go.Bar(x=scores.index,
                y=scores['4'],
                name='4 star',
                marker_color='yellow',
                hovertext = round(scores['%4'],3)
                ))
fig.add_trace(go.Bar(x=scores.index,
                y=scores['5'],
                name='5 star',
                marker_color='green',
                hovertext = round(scores['%5'],3)
                ))
fig.update_layout(
    title='Breakdown of Daily Ratings',
    #xaxis_tickfont_size=14,
    #yaxis=dict(
    #    title='USD (millions)',
    #    titlefont_size=16,
    #    tickfont_size=14,
    #),
    #legend=dict(
    #    x=0,
    #    y=1.0,
    #    bgcolor='rgba(255, 255, 255, 0)',
    #    bordercolor='rgba(255, 255, 255, 0)'
    #),
    barmode='stack',
    bargap=0.15, # gap between bars of adjacent location coordinates.
    bargroupgap=0.1 # gap between bars of the same location coordinate.
)
fig.show()
#fig.write_image('sentiment_graph1.png')


## Sentiment analysis
def extract_features(word_list):
    return dict([(word, True) for word in word_list])
if __name__ == '__main__':
    positive_fileids = movie_reviews.fileids('pos')
    negative_fileids = movie_reviews.fileids('neg')


features_positive = [(extract_features(movie_reviews.words(fileids=[f])), 
                      'Positive') for f in positive_fileids]
features_negative = [(extract_features(movie_reviews.words(fileids=[f])),
                         'Negative') for f in negative_fileids]


threshold_factor = 0.2
threshold_positive = int(threshold_factor * len(features_positive))
threshold_negative = int(threshold_factor * len(features_negative))


features_train = features_positive[:threshold_positive] + features_negative[:threshold_negative]
features_test = features_positive[threshold_positive:] + features_negative[threshold_negative:]  
print ("\nNumber of training datapoints:", len(features_train))
print ("Number of test datapoints:", len(features_test))



classifier = NaiveBayesClassifier.train(features_train)
print ("\nAccuracy of the classifier:", nltk.classify.util.accuracy(classifier, features_test))



print ("\nTop 10 most informative words:")
for item in classifier.most_informative_features()[:10]:
    print( item[0])


#reviews = pd.read_csv('iosAllReviews.csv')


input_reviews = reviews['text']

print ("\nPredictions:")
pred_sent = []
prob_sent = []
for review in input_reviews:
    #print ("\nReview:", review)
    probdist = classifier.prob_classify(extract_features(review.split()))
    pred_sentiment = probdist.max()
    pred_sent.append(pred_sentiment)
    prob_sent.append(round(probdist.prob(pred_sentiment), 2))

print ("Predicted sentiment:", pred_sentiment )
print ("Probability:", round(probdist.prob(pred_sentiment), 2))


sent_graph = pd.DataFrame(pred_sent, columns=['sentiment'])
sent_graph['probability'] = prob_sent

sent_graph.sentiment[sent_graph.sentiment == 'Negative'] = -1
sent_graph.sentiment[sent_graph.sentiment == 'Positive'] = 1

sent_graph['vals'] = sent_graph['sentiment']*sent_graph['probability']
sent_graph['date'] = reviews['date']

x = np.arange(len(scores['1']), dtype=int)
x = np.full_like(x, 1)
scores['sent'] = x
for day, group in byday:
    #print(group.vals.mean(),day)
    scores.loc[[day],['sent']] = group.vals.mean()

sent1 = go.Figure()
sent1.add_trace(go.Bar(x=dates, y=scores['sent'],
                marker_color='crimson',
                name='sentiment'))
sent1.show()

## Frequent words --- you should have the new code
counts = dict()

for review in reviews['text']:
    words = review.split()
    for word in words:
        if word in counts:
            counts[word] += 1
        else:
            counts[word] = 1


new_dict = dict()
for item in counts.items():
    if item[1] >= 1000:
        new_dict[item[0]] = item[1]

%matplotlib inline
pl = plt.bar(range(len(new_dict)),list(new_dict.values()),align='center')
plt.xticks(range(len(new_dict)),list(new_dict.keys()))
plt.show()

## Create a pdf
#output graph in a pdf
pdf = FPDF()
pdf.add_page()
pdf.set_xy(0, 0)
pdf.set_font('arial', 'B', 12)
pdf.cell(w=50,h=20,txt='Sentiment analysis of ' + app_name)
#pdf.cell(90, 30, " ", 0, 2, 'C')
pdf.image('sentiment_graph1.png', x = 5, y = 15, w = 200, h = 0, type = '', link = '')
pdf.output('test.pdf', 'F')
pdf.addpage()
