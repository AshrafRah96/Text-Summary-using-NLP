## Simple Text Summary Model Using NLP
##### Ashraf Rahman

What is NLP?
- NLP also is known as '***Natural language processing***' is a field in machine learning which is used to ***understand, analyse and generate*** natural human texts.

What is Text Summary?
- Text Summary allows the model to extract unbiased key insights and information to generate a shorted summary of the document.
- There are two types of summarization in particular:
	- **Extractive**: Investigates important sentences using a ranking system and picks the sentences to be used in the summary. (Used in this example)
	- **Abstractive**: Similar to extractive however it paraphrases and achieves a more human-like summary. 

Use Cases for NLP? (To name a few)
- Text Summary.
- Chat-Bot for first-line support
- Open/Closed domain Q&A. 
- Automating and making sense of unstructured data.
- Speech-To-Text for writing notes, resulting in more focus towards meetings.

How it’s done? <br />
1) Tokenisation. <br />
&nbsp;&nbsp; a) Word Tokenisation. <br />
&nbsp;&nbsp; b) Sentence Tokenisation. <br />
3) TF-IDF (Text Frequency-Inverse Document Frequency): Identify the importance of words and sentences, using a frequency table. <br />
4) Normalisation. <br />
5) Summerisation. <br />

> **Note**: 
Tokenisation: Breaking the raw text into small chunks. Tokenization breaks the raw text into words, sentences called tokens. These tokens help in understanding the context or developing the model for the NLP. The tokenization helps in interpreting the meaning of the text by analyzing the sequence of the words.


```python
# Article I will be using for testing

text = """
What is Artifical intelligence? (https://www.bbc.co.uk/newsround/49274918)
Artificial intelligence - or AI for short - is technology that enables a computer to think or act in a more 'human' way. It does this by taking in information from its surroundings, and deciding its response based on what it learns or senses.
It affects the the way we live, work and have fun in our spare time - and sometimes without us even realising.
AI is becoming a bigger part of our lives, as the technology behind it becomes more and more advanced. Machines are improving their ability to 'learn' from mistakes and change how they approach a task the next time they try it.
Some researchers are even trying to teach robots about feelings and emotions.
You might not realise some of the devices and daily activities which rely on AI technology - phones, video games and going shopping, for example.
More technology
Why did this photo make history 60 years ago?
How robots and drones are changing deliveries
Flyboard inventor crosses English Channel
Why Instagram is going to hide your 'likes'
Some people think that the technology is a really good idea, while others aren't so sure.
Just this month, it was announced that the NHS in England is setting up a special AI laboratory to boost the role of AI within the health service.
Announcing that the government will spend £250 million on this, Health Secretary Matt Hancock said the technology had "enormous power" to improve care, save lives and ensure doctors had more time to spend with patients.
Read on to find out more about AI and let us know what you think about it in the comments below.
What does AI do?
AI can be used for many different tasks and activities.
Personal electronic devices or accounts (like our phones or social media) use AI to learn more about us and the things that we like. One example of this is entertainment services like Netflix which use the technology to understand what we like to watch and recommend other shows based on what they learn.
It can make video games more challenging by studying how a player behaves, while home assistants like Alexa and Siri also rely on it.
It has been announced that NHS England will spend millions on AI in order to improve patient care and research
AI can be used in healthcare, not only for research purposes, but also to take better care of patients through improved diagnosis and monitoring.
It also has uses within transport too. For example, driverless cars are an example of AI tech in action, while it is used extensively in the aviation industry (for example, in flight simulators).
Farmers can use AI to monitor crops and conditions, and to make predictions, which will help them to be more efficient.
You only have to look at what some of these AI robots can do to see just how advanced the technology is and imagine many other jobs for which it could be used.
Where did AI come from?
The term 'artificial intelligence' was first used in 1956.
In the 1960s, scientists were teaching computers how to mimic - or copy - human decision-making.
This developed into research around 'machine learning', in which robots were taught to learn for themselves and remember their mistakes, instead of simply copying. Algorithms play a big part in machine learning as they help computers and robots to know what to do.
What is an algorithm?
An algorithm is basically a set of rules or instructions which a computer can use to help solve a problem or come to a decision about what to do next.
From here, the research has continued to develop, with scientists now exploring 'machine perception'. This involves giving machines and robots special sensors to help them to see, hear, feel and taste things like human do - and adjust how they behave as a result of what they sense.
The idea is that the more this technology develops, the more robots will be able to 'understand' and read situations, and determine their response as a result of the information that they pick up.
Why are people worried about AI?
Many people have concerns about AI technology and teaching robots too much.
Famous scientist Sir Stephen Hawking spoke out about it in the past. He said that although the AI we've made so far has been very useful and helpful, he worried that if we teach robots too much, they could become smarter than humans and potentially cause problems.
Sir Stephen Hawking spoke out about AI and said that he had concerns that the technology could cause problems in the future
People have expressed concerns about privacy too. For example, critics think that it could become a problem if AI learns too much about what we like to look at online and encourages us to spend too much time on electronic devices.
Another concern about AI is that if robots and computers become very intelligent, they could learn to do jobs which people would usually have to do, which could leave some people unemployed.
Other people disagree, saying that the technology will never be as advanced as human thoughts and actions, so there is not a danger of robots 'taking over' in the way that some critics have described.
What do you think about AI? Do you think that it is a good thing or a bad thing? Let us know in the comments below.
"""
```


```python
import spacy
from spacy.lang.en import English
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
```

>Note:
Stop Words: Most Common words

We remove stop words and punctuation as they are common and do not contribute to the investigation of important sentences.


```python
stopwords = list(STOP_WORDS)
print(stopwords)
```

    ['’s', 'nor', 'whenever', 'much', 'least', 'about', 'before', 'throughout', 'may', 'beside', 'had', 'perhaps', 'four', 'through', 'yourselves', 'everything', '‘ll', 'rather', 'those', 'into', 'bottom', 'take', 'sometime', 'herein', 'amount', 'n‘t', 'cannot', 'be', 'hence', 'indeed', 'twelve', 'whose', 'do', 'among', 'hereupon', 'mostly', 'against', 'two', 'various', 'around', 'thereafter', 'whole', 'still', 'either', 'no', 'besides', 'become', 'somehow', 'ca', 'us', 'due', 'often', 'mine', 'will', 'used', 'did', 'first', 'made', 'never', 'on', 'because', 'except', 'just', 'themselves', 'anyway', 'n’t', 'herself', 're', 'me', 'thereupon', 'how', 'from', 'part', 'please', 'put', 'always', 'became', 'across', 'wherever', 'whether', '‘s', 'my', 'show', 'something', 'also', 'at', 'one', 'you', 'both', 'these', '‘d', 'could', 'it', 'meanwhile', 'say', 'under', 'using', 'that', 'if', 'would', '‘re', 'off', "'m", 'onto', 'call', 'until', 'his', 'once', 'whither', 'and', 'thru', 'hundred', 'above', 'but', 'whatever', 'each', 'nowhere', 'everyone', 'itself', 'otherwise', 'twenty', 'by', 'down', 'so', 'via', 'keep', 'over', 'upon', 'as', 'last', 'elsewhere', 'up', 'eight', 'name', 'below', 'seeming', 'have', 'afterwards', 'neither', 'our', 'since', "'d", 'therein', 'sixty', 'nobody', 'three', 'empty', 'though', 'anywhere', "'re", 'can', 'formerly', 'are', 'most', 'many', '’m', 'might', 'quite', 'side', 'everywhere', 'without', 'see', 'back', 'enough', 'whereas', 'further', 'own', 'its', 'he', 'now', 'while', 'done', 'third', 'am', 'another', 'an', 'becomes', 'together', 'after', 'hereby', 'per', 'seems', 'within', 'well', 'give', 'ever', 'moreover', 'to', 'six', "'ll", 'few', 'becoming', '‘m', 'them', 'five', 'every', 'thus', 'along', 'why', 'is', 'whereupon', 'next', 'such', 'nevertheless', 'for', 'what', 'here', 'move', 'serious', '‘ve', 'alone', 'has', 'latter', 'forty', 'somewhere', 'thence', 'other', 'too', 'the', 'hereafter', 'less', 'whereafter', 'we', "'s", 'even', 'then', 'get', 'nine', 'else', 'some', 'this', 'already', 'she', 'must', 'being', 'of', 'was', 'ourselves', 'where', 'fifteen', 'although', 'him', 'there', 'anyhow', 'which', 'yours', 'regarding', 'amongst', 'yourself', '’ll', 'thereby', 'whoever', 'full', 'doing', 'your', 'eleven', 'or', 'out', 'anything', 'not', 'when', 'ten', 'beyond', 'therefore', 'unless', 'whereby', 'same', 'toward', '’re', 'go', '’d', 'wherein', "n't", 'a', 'seem', 'they', 'whom', 'her', 'several', 'again', 'myself', 'been', 'yet', 'however', 'anyone', 'top', 'namely', 'others', 'really', 'i', 'himself', 'should', 'between', 'their', 'all', 'only', 'seemed', 'whence', 'sometimes', 'almost', 'make', '’ve', 'front', 'none', 'very', 'during', 'nothing', 'who', 'former', 'in', 'hers', 'behind', 'latterly', "'ve", 'were', 'more', 'someone', 'any', 'noone', 'than', 'fifty', 'does', 'beforehand', 'ours', 'towards', 'with']
    


```python
punctuation += "\n"
punctuation += "\n\n"

punctuation
```




    '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~\n\n\n'



#### Loading the model


```python
nlp = spacy.load('en')
```

#### Tokenisation


```python
doc = nlp(text) # Loading the text into the nlp model 
tokens = [token.text for token in doc] # Extracting the text to a list
print(tokens)
```

    ['\n', 'What', 'is', 'Artifical', 'intelligence', '?', '(', 'https://www.bbc.co.uk/newsround/49274918', ')', '\n', 'Artificial', 'intelligence', '-', 'or', 'AI', 'for', 'short', '-', 'is', 'technology', 'that', 'enables', 'a', 'computer', 'to', 'think', 'or', 'act', 'in', 'a', 'more', "'", 'human', "'", 'way', '.', 'It', 'does', 'this', 'by', 'taking', 'in', 'information', 'from', 'its', 'surroundings', ',', 'and', 'deciding', 'its', 'response', 'based', 'on', 'what', 'it', 'learns', 'or', 'senses', '.', '\n', 'It', 'affects', 'the', 'the', 'way', 'we', 'live', ',', 'work', 'and', 'have', 'fun', 'in', 'our', 'spare', 'time', '-', 'and', 'sometimes', 'without', 'us', 'even', 'realising', '.', '\n', 'AI', 'is', 'becoming', 'a', 'bigger', 'part', 'of', 'our', 'lives', ',', 'as', 'the', 'technology', 'behind', 'it', 'becomes', 'more', 'and', 'more', 'advanced', '.', 'Machines', 'are', 'improving', 'their', 'ability', 'to', "'", 'learn', "'", 'from', 'mistakes', 'and', 'change', 'how', 'they', 'approach', 'a', 'task', 'the', 'next', 'time', 'they', 'try', 'it', '.', '\n', 'Some', 'researchers', 'are', 'even', 'trying', 'to', 'teach', 'robots', 'about', 'feelings', 'and', 'emotions', '.', '\n', 'You', 'might', 'not', 'realise', 'some', 'of', 'the', 'devices', 'and', 'daily', 'activities', 'which', 'rely', 'on', 'AI', 'technology', '-', 'phones', ',', 'video', 'games', 'and', 'going', 'shopping', ',', 'for', 'example', '.', '\n', 'More', 'technology', '\n', 'Why', 'did', 'this', 'photo', 'make', 'history', '60', 'years', 'ago', '?', '\n', 'How', 'robots', 'and', 'drones', 'are', 'changing', 'deliveries', '\n', 'Flyboard', 'inventor', 'crosses', 'English', 'Channel', '\n', 'Why', 'Instagram', 'is', 'going', 'to', 'hide', 'your', "'", 'likes', "'", '\n', 'Some', 'people', 'think', 'that', 'the', 'technology', 'is', 'a', 'really', 'good', 'idea', ',', 'while', 'others', 'are', "n't", 'so', 'sure', '.', '\n', 'Just', 'this', 'month', ',', 'it', 'was', 'announced', 'that', 'the', 'NHS', 'in', 'England', 'is', 'setting', 'up', 'a', 'special', 'AI', 'laboratory', 'to', 'boost', 'the', 'role', 'of', 'AI', 'within', 'the', 'health', 'service', '.', '\n', 'Announcing', 'that', 'the', 'government', 'will', 'spend', '£', '250', 'million', 'on', 'this', ',', 'Health', 'Secretary', 'Matt', 'Hancock', 'said', 'the', 'technology', 'had', '"', 'enormous', 'power', '"', 'to', 'improve', 'care', ',', 'save', 'lives', 'and', 'ensure', 'doctors', 'had', 'more', 'time', 'to', 'spend', 'with', 'patients', '.', '\n', 'Read', 'on', 'to', 'find', 'out', 'more', 'about', 'AI', 'and', 'let', 'us', 'know', 'what', 'you', 'think', 'about', 'it', 'in', 'the', 'comments', 'below', '.', '\n', 'What', 'does', 'AI', 'do', '?', '\n', 'AI', 'can', 'be', 'used', 'for', 'many', 'different', 'tasks', 'and', 'activities', '.', '\n', 'Personal', 'electronic', 'devices', 'or', 'accounts', '(', 'like', 'our', 'phones', 'or', 'social', 'media', ')', 'use', 'AI', 'to', 'learn', 'more', 'about', 'us', 'and', 'the', 'things', 'that', 'we', 'like', '.', 'One', 'example', 'of', 'this', 'is', 'entertainment', 'services', 'like', 'Netflix', 'which', 'use', 'the', 'technology', 'to', 'understand', 'what', 'we', 'like', 'to', 'watch', 'and', 'recommend', 'other', 'shows', 'based', 'on', 'what', 'they', 'learn', '.', '\n', 'It', 'can', 'make', 'video', 'games', 'more', 'challenging', 'by', 'studying', 'how', 'a', 'player', 'behaves', ',', 'while', 'home', 'assistants', 'like', 'Alexa', 'and', 'Siri', 'also', 'rely', 'on', 'it', '.', '\n', 'It', 'has', 'been', 'announced', 'that', 'NHS', 'England', 'will', 'spend', 'millions', 'on', 'AI', 'in', 'order', 'to', 'improve', 'patient', 'care', 'and', 'research', '\n', 'AI', 'can', 'be', 'used', 'in', 'healthcare', ',', 'not', 'only', 'for', 'research', 'purposes', ',', 'but', 'also', 'to', 'take', 'better', 'care', 'of', 'patients', 'through', 'improved', 'diagnosis', 'and', 'monitoring', '.', '\n', 'It', 'also', 'has', 'uses', 'within', 'transport', 'too', '.', 'For', 'example', ',', 'driverless', 'cars', 'are', 'an', 'example', 'of', 'AI', 'tech', 'in', 'action', ',', 'while', 'it', 'is', 'used', 'extensively', 'in', 'the', 'aviation', 'industry', '(', 'for', 'example', ',', 'in', 'flight', 'simulators', ')', '.', '\n', 'Farmers', 'can', 'use', 'AI', 'to', 'monitor', 'crops', 'and', 'conditions', ',', 'and', 'to', 'make', 'predictions', ',', 'which', 'will', 'help', 'them', 'to', 'be', 'more', 'efficient', '.', '\n', 'You', 'only', 'have', 'to', 'look', 'at', 'what', 'some', 'of', 'these', 'AI', 'robots', 'can', 'do', 'to', 'see', 'just', 'how', 'advanced', 'the', 'technology', 'is', 'and', 'imagine', 'many', 'other', 'jobs', 'for', 'which', 'it', 'could', 'be', 'used', '.', '\n', 'Where', 'did', 'AI', 'come', 'from', '?', '\n', 'The', 'term', "'", 'artificial', 'intelligence', "'", 'was', 'first', 'used', 'in', '1956', '.', '\n', 'In', 'the', '1960s', ',', 'scientists', 'were', 'teaching', 'computers', 'how', 'to', 'mimic', '-', 'or', 'copy', '-', 'human', 'decision', '-', 'making', '.', '\n', 'This', 'developed', 'into', 'research', 'around', "'", 'machine', 'learning', "'", ',', 'in', 'which', 'robots', 'were', 'taught', 'to', 'learn', 'for', 'themselves', 'and', 'remember', 'their', 'mistakes', ',', 'instead', 'of', 'simply', 'copying', '.', 'Algorithms', 'play', 'a', 'big', 'part', 'in', 'machine', 'learning', 'as', 'they', 'help', 'computers', 'and', 'robots', 'to', 'know', 'what', 'to', 'do', '.', '\n', 'What', 'is', 'an', 'algorithm', '?', '\n', 'An', 'algorithm', 'is', 'basically', 'a', 'set', 'of', 'rules', 'or', 'instructions', 'which', 'a', 'computer', 'can', 'use', 'to', 'help', 'solve', 'a', 'problem', 'or', 'come', 'to', 'a', 'decision', 'about', 'what', 'to', 'do', 'next', '.', '\n', 'From', 'here', ',', 'the', 'research', 'has', 'continued', 'to', 'develop', ',', 'with', 'scientists', 'now', 'exploring', "'", 'machine', 'perception', "'", '.', 'This', 'involves', 'giving', 'machines', 'and', 'robots', 'special', 'sensors', 'to', 'help', 'them', 'to', 'see', ',', 'hear', ',', 'feel', 'and', 'taste', 'things', 'like', 'human', 'do', '-', 'and', 'adjust', 'how', 'they', 'behave', 'as', 'a', 'result', 'of', 'what', 'they', 'sense', '.', '\n', 'The', 'idea', 'is', 'that', 'the', 'more', 'this', 'technology', 'develops', ',', 'the', 'more', 'robots', 'will', 'be', 'able', 'to', "'", 'understand', "'", 'and', 'read', 'situations', ',', 'and', 'determine', 'their', 'response', 'as', 'a', 'result', 'of', 'the', 'information', 'that', 'they', 'pick', 'up', '.', '\n', 'Why', 'are', 'people', 'worried', 'about', 'AI', '?', '\n', 'Many', 'people', 'have', 'concerns', 'about', 'AI', 'technology', 'and', 'teaching', 'robots', 'too', 'much', '.', '\n', 'Famous', 'scientist', 'Sir', 'Stephen', 'Hawking', 'spoke', 'out', 'about', 'it', 'in', 'the', 'past', '.', 'He', 'said', 'that', 'although', 'the', 'AI', 'we', "'ve", 'made', 'so', 'far', 'has', 'been', 'very', 'useful', 'and', 'helpful', ',', 'he', 'worried', 'that', 'if', 'we', 'teach', 'robots', 'too', 'much', ',', 'they', 'could', 'become', 'smarter', 'than', 'humans', 'and', 'potentially', 'cause', 'problems', '.', '\n', 'Sir', 'Stephen', 'Hawking', 'spoke', 'out', 'about', 'AI', 'and', 'said', 'that', 'he', 'had', 'concerns', 'that', 'the', 'technology', 'could', 'cause', 'problems', 'in', 'the', 'future', '\n', 'People', 'have', 'expressed', 'concerns', 'about', 'privacy', 'too', '.', 'For', 'example', ',', 'critics', 'think', 'that', 'it', 'could', 'become', 'a', 'problem', 'if', 'AI', 'learns', 'too', 'much', 'about', 'what', 'we', 'like', 'to', 'look', 'at', 'online', 'and', 'encourages', 'us', 'to', 'spend', 'too', 'much', 'time', 'on', 'electronic', 'devices', '.', '\n', 'Another', 'concern', 'about', 'AI', 'is', 'that', 'if', 'robots', 'and', 'computers', 'become', 'very', 'intelligent', ',', 'they', 'could', 'learn', 'to', 'do', 'jobs', 'which', 'people', 'would', 'usually', 'have', 'to', 'do', ',', 'which', 'could', 'leave', 'some', 'people', 'unemployed', '.', '\n', 'Other', 'people', 'disagree', ',', 'saying', 'that', 'the', 'technology', 'will', 'never', 'be', 'as', 'advanced', 'as', 'human', 'thoughts', 'and', 'actions', ',', 'so', 'there', 'is', 'not', 'a', 'danger', 'of', 'robots', "'", 'taking', 'over', "'", 'in', 'the', 'way', 'that', 'some', 'critics', 'have', 'described', '.', '\n', 'What', 'do', 'you', 'think', 'about', 'AI', '?', 'Do', 'you', 'think', 'that', 'it', 'is', 'a', 'good', 'thing', 'or', 'a', 'bad', 'thing', '?', 'Let', 'us', 'know', 'in', 'the', 'comments', 'below', '.', '\n']
    

#### TF-IDF (Generating a frequency table)


```python
word_frequencies = {}
for word in doc:
    
    if word.text.lower() not in stopwords:
        if word.text.lower() not in punctuation:
            
            if word.text not in word_frequencies:
                word_frequencies[word.text] = 1
            else:
                word_frequencies[word.text] += 1

word_frequencies
```




    {'Artifical': 1,
     'intelligence': 3,
     'https://www.bbc.co.uk/newsround/49274918': 1,
     'Artificial': 1,
     'AI': 22,
     'short': 1,
     'technology': 12,
     'enables': 1,
     'computer': 2,
     'think': 6,
     'act': 1,
     'human': 4,
     'way': 3,
     'taking': 2,
     'information': 2,
     'surroundings': 1,
     'deciding': 1,
     'response': 2,
     'based': 2,
     'learns': 2,
     'senses': 1,
     'affects': 1,
     'live': 1,
     'work': 1,
     'fun': 1,
     'spare': 1,
     'time': 4,
     'realising': 1,
     'bigger': 1,
     'lives': 2,
     'advanced': 3,
     'Machines': 1,
     'improving': 1,
     'ability': 1,
     'learn': 5,
     'mistakes': 2,
     'change': 1,
     'approach': 1,
     'task': 1,
     'try': 1,
     'researchers': 1,
     'trying': 1,
     'teach': 2,
     'robots': 11,
     'feelings': 1,
     'emotions': 1,
     'realise': 1,
     'devices': 3,
     'daily': 1,
     'activities': 2,
     'rely': 2,
     'phones': 2,
     'video': 2,
     'games': 2,
     'going': 2,
     'shopping': 1,
     'example': 6,
     'photo': 1,
     'history': 1,
     '60': 1,
     'years': 1,
     'ago': 1,
     'drones': 1,
     'changing': 1,
     'deliveries': 1,
     'Flyboard': 1,
     'inventor': 1,
     'crosses': 1,
     'English': 1,
     'Channel': 1,
     'Instagram': 1,
     'hide': 1,
     'likes': 1,
     'people': 6,
     'good': 2,
     'idea': 2,
     'sure': 1,
     'month': 1,
     'announced': 2,
     'NHS': 2,
     'England': 2,
     'setting': 1,
     'special': 2,
     'laboratory': 1,
     'boost': 1,
     'role': 1,
     'health': 1,
     'service': 1,
     'Announcing': 1,
     'government': 1,
     'spend': 4,
     '£': 1,
     '250': 1,
     'million': 1,
     'Health': 1,
     'Secretary': 1,
     'Matt': 1,
     'Hancock': 1,
     'said': 3,
     'enormous': 1,
     'power': 1,
     'improve': 2,
     'care': 3,
     'save': 1,
     'ensure': 1,
     'doctors': 1,
     'patients': 2,
     'Read': 1,
     'find': 1,
     'let': 1,
     'know': 3,
     'comments': 2,
     'different': 1,
     'tasks': 1,
     'Personal': 1,
     'electronic': 2,
     'accounts': 1,
     'like': 7,
     'social': 1,
     'media': 1,
     'use': 4,
     'things': 2,
     'entertainment': 1,
     'services': 1,
     'Netflix': 1,
     'understand': 2,
     'watch': 1,
     'recommend': 1,
     'shows': 1,
     'challenging': 1,
     'studying': 1,
     'player': 1,
     'behaves': 1,
     'home': 1,
     'assistants': 1,
     'Alexa': 1,
     'Siri': 1,
     'millions': 1,
     'order': 1,
     'patient': 1,
     'research': 4,
     'healthcare': 1,
     'purposes': 1,
     'better': 1,
     'improved': 1,
     'diagnosis': 1,
     'monitoring': 1,
     'uses': 1,
     'transport': 1,
     'driverless': 1,
     'cars': 1,
     'tech': 1,
     'action': 1,
     'extensively': 1,
     'aviation': 1,
     'industry': 1,
     'flight': 1,
     'simulators': 1,
     'Farmers': 1,
     'monitor': 1,
     'crops': 1,
     'conditions': 1,
     'predictions': 1,
     'help': 4,
     'efficient': 1,
     'look': 2,
     'imagine': 1,
     'jobs': 2,
     'come': 2,
     'term': 1,
     'artificial': 1,
     '1956': 1,
     '1960s': 1,
     'scientists': 2,
     'teaching': 2,
     'computers': 3,
     'mimic': 1,
     'copy': 1,
     'decision': 2,
     'making': 1,
     'developed': 1,
     'machine': 3,
     'learning': 2,
     'taught': 1,
     'remember': 1,
     'instead': 1,
     'simply': 1,
     'copying': 1,
     'Algorithms': 1,
     'play': 1,
     'big': 1,
     'algorithm': 2,
     'basically': 1,
     'set': 1,
     'rules': 1,
     'instructions': 1,
     'solve': 1,
     'problem': 2,
     'continued': 1,
     'develop': 1,
     'exploring': 1,
     'perception': 1,
     'involves': 1,
     'giving': 1,
     'machines': 1,
     'sensors': 1,
     'hear': 1,
     'feel': 1,
     'taste': 1,
     'adjust': 1,
     'behave': 1,
     'result': 2,
     'sense': 1,
     'develops': 1,
     'able': 1,
     'read': 1,
     'situations': 1,
     'determine': 1,
     'pick': 1,
     'worried': 2,
     'concerns': 3,
     'Famous': 1,
     'scientist': 1,
     'Sir': 2,
     'Stephen': 2,
     'Hawking': 2,
     'spoke': 2,
     'past': 1,
     'far': 1,
     'useful': 1,
     'helpful': 1,
     'smarter': 1,
     'humans': 1,
     'potentially': 1,
     'cause': 2,
     'problems': 2,
     'future': 1,
     'People': 1,
     'expressed': 1,
     'privacy': 1,
     'critics': 2,
     'online': 1,
     'encourages': 1,
     'concern': 1,
     'intelligent': 1,
     'usually': 1,
     'leave': 1,
     'unemployed': 1,
     'disagree': 1,
     'saying': 1,
     'thoughts': 1,
     'actions': 1,
     'danger': 1,
     'described': 1,
     'thing': 2,
     'bad': 1,
     'Let': 1}




```python
max_frequency = max(word_frequencies.values())
max_frequency
```




    22



#### We must normalise the values so it can improve the performance of the model by reducing the values to a range. (Although it is not required in this example it is good practice to implement normalisation for advanced models e.g. Abstractive Text Summary)


```python
for word in word_frequencies.keys():
    word_frequencies[word] /= max_frequency
    
word_frequencies
```




    {'Artifical': 0.045454545454545456,
     'intelligence': 0.13636363636363635,
     'https://www.bbc.co.uk/newsround/49274918': 0.045454545454545456,
     'Artificial': 0.045454545454545456,
     'AI': 1.0,
     'short': 0.045454545454545456,
     'technology': 0.5454545454545454,
     'enables': 0.045454545454545456,
     'computer': 0.09090909090909091,
     'think': 0.2727272727272727,
     'act': 0.045454545454545456,
     'human': 0.18181818181818182,
     'way': 0.13636363636363635,
     'taking': 0.09090909090909091,
     'information': 0.09090909090909091,
     'surroundings': 0.045454545454545456,
     'deciding': 0.045454545454545456,
     'response': 0.09090909090909091,
     'based': 0.09090909090909091,
     'learns': 0.09090909090909091,
     'senses': 0.045454545454545456,
     'affects': 0.045454545454545456,
     'live': 0.045454545454545456,
     'work': 0.045454545454545456,
     'fun': 0.045454545454545456,
     'spare': 0.045454545454545456,
     'time': 0.18181818181818182,
     'realising': 0.045454545454545456,
     'bigger': 0.045454545454545456,
     'lives': 0.09090909090909091,
     'advanced': 0.13636363636363635,
     'Machines': 0.045454545454545456,
     'improving': 0.045454545454545456,
     'ability': 0.045454545454545456,
     'learn': 0.22727272727272727,
     'mistakes': 0.09090909090909091,
     'change': 0.045454545454545456,
     'approach': 0.045454545454545456,
     'task': 0.045454545454545456,
     'try': 0.045454545454545456,
     'researchers': 0.045454545454545456,
     'trying': 0.045454545454545456,
     'teach': 0.09090909090909091,
     'robots': 0.5,
     'feelings': 0.045454545454545456,
     'emotions': 0.045454545454545456,
     'realise': 0.045454545454545456,
     'devices': 0.13636363636363635,
     'daily': 0.045454545454545456,
     'activities': 0.09090909090909091,
     'rely': 0.09090909090909091,
     'phones': 0.09090909090909091,
     'video': 0.09090909090909091,
     'games': 0.09090909090909091,
     'going': 0.09090909090909091,
     'shopping': 0.045454545454545456,
     'example': 0.2727272727272727,
     'photo': 0.045454545454545456,
     'history': 0.045454545454545456,
     '60': 0.045454545454545456,
     'years': 0.045454545454545456,
     'ago': 0.045454545454545456,
     'drones': 0.045454545454545456,
     'changing': 0.045454545454545456,
     'deliveries': 0.045454545454545456,
     'Flyboard': 0.045454545454545456,
     'inventor': 0.045454545454545456,
     'crosses': 0.045454545454545456,
     'English': 0.045454545454545456,
     'Channel': 0.045454545454545456,
     'Instagram': 0.045454545454545456,
     'hide': 0.045454545454545456,
     'likes': 0.045454545454545456,
     'people': 0.2727272727272727,
     'good': 0.09090909090909091,
     'idea': 0.09090909090909091,
     'sure': 0.045454545454545456,
     'month': 0.045454545454545456,
     'announced': 0.09090909090909091,
     'NHS': 0.09090909090909091,
     'England': 0.09090909090909091,
     'setting': 0.045454545454545456,
     'special': 0.09090909090909091,
     'laboratory': 0.045454545454545456,
     'boost': 0.045454545454545456,
     'role': 0.045454545454545456,
     'health': 0.045454545454545456,
     'service': 0.045454545454545456,
     'Announcing': 0.045454545454545456,
     'government': 0.045454545454545456,
     'spend': 0.18181818181818182,
     '£': 0.045454545454545456,
     '250': 0.045454545454545456,
     'million': 0.045454545454545456,
     'Health': 0.045454545454545456,
     'Secretary': 0.045454545454545456,
     'Matt': 0.045454545454545456,
     'Hancock': 0.045454545454545456,
     'said': 0.13636363636363635,
     'enormous': 0.045454545454545456,
     'power': 0.045454545454545456,
     'improve': 0.09090909090909091,
     'care': 0.13636363636363635,
     'save': 0.045454545454545456,
     'ensure': 0.045454545454545456,
     'doctors': 0.045454545454545456,
     'patients': 0.09090909090909091,
     'Read': 0.045454545454545456,
     'find': 0.045454545454545456,
     'let': 0.045454545454545456,
     'know': 0.13636363636363635,
     'comments': 0.09090909090909091,
     'different': 0.045454545454545456,
     'tasks': 0.045454545454545456,
     'Personal': 0.045454545454545456,
     'electronic': 0.09090909090909091,
     'accounts': 0.045454545454545456,
     'like': 0.3181818181818182,
     'social': 0.045454545454545456,
     'media': 0.045454545454545456,
     'use': 0.18181818181818182,
     'things': 0.09090909090909091,
     'entertainment': 0.045454545454545456,
     'services': 0.045454545454545456,
     'Netflix': 0.045454545454545456,
     'understand': 0.09090909090909091,
     'watch': 0.045454545454545456,
     'recommend': 0.045454545454545456,
     'shows': 0.045454545454545456,
     'challenging': 0.045454545454545456,
     'studying': 0.045454545454545456,
     'player': 0.045454545454545456,
     'behaves': 0.045454545454545456,
     'home': 0.045454545454545456,
     'assistants': 0.045454545454545456,
     'Alexa': 0.045454545454545456,
     'Siri': 0.045454545454545456,
     'millions': 0.045454545454545456,
     'order': 0.045454545454545456,
     'patient': 0.045454545454545456,
     'research': 0.18181818181818182,
     'healthcare': 0.045454545454545456,
     'purposes': 0.045454545454545456,
     'better': 0.045454545454545456,
     'improved': 0.045454545454545456,
     'diagnosis': 0.045454545454545456,
     'monitoring': 0.045454545454545456,
     'uses': 0.045454545454545456,
     'transport': 0.045454545454545456,
     'driverless': 0.045454545454545456,
     'cars': 0.045454545454545456,
     'tech': 0.045454545454545456,
     'action': 0.045454545454545456,
     'extensively': 0.045454545454545456,
     'aviation': 0.045454545454545456,
     'industry': 0.045454545454545456,
     'flight': 0.045454545454545456,
     'simulators': 0.045454545454545456,
     'Farmers': 0.045454545454545456,
     'monitor': 0.045454545454545456,
     'crops': 0.045454545454545456,
     'conditions': 0.045454545454545456,
     'predictions': 0.045454545454545456,
     'help': 0.18181818181818182,
     'efficient': 0.045454545454545456,
     'look': 0.09090909090909091,
     'imagine': 0.045454545454545456,
     'jobs': 0.09090909090909091,
     'come': 0.09090909090909091,
     'term': 0.045454545454545456,
     'artificial': 0.045454545454545456,
     '1956': 0.045454545454545456,
     '1960s': 0.045454545454545456,
     'scientists': 0.09090909090909091,
     'teaching': 0.09090909090909091,
     'computers': 0.13636363636363635,
     'mimic': 0.045454545454545456,
     'copy': 0.045454545454545456,
     'decision': 0.09090909090909091,
     'making': 0.045454545454545456,
     'developed': 0.045454545454545456,
     'machine': 0.13636363636363635,
     'learning': 0.09090909090909091,
     'taught': 0.045454545454545456,
     'remember': 0.045454545454545456,
     'instead': 0.045454545454545456,
     'simply': 0.045454545454545456,
     'copying': 0.045454545454545456,
     'Algorithms': 0.045454545454545456,
     'play': 0.045454545454545456,
     'big': 0.045454545454545456,
     'algorithm': 0.09090909090909091,
     'basically': 0.045454545454545456,
     'set': 0.045454545454545456,
     'rules': 0.045454545454545456,
     'instructions': 0.045454545454545456,
     'solve': 0.045454545454545456,
     'problem': 0.09090909090909091,
     'continued': 0.045454545454545456,
     'develop': 0.045454545454545456,
     'exploring': 0.045454545454545456,
     'perception': 0.045454545454545456,
     'involves': 0.045454545454545456,
     'giving': 0.045454545454545456,
     'machines': 0.045454545454545456,
     'sensors': 0.045454545454545456,
     'hear': 0.045454545454545456,
     'feel': 0.045454545454545456,
     'taste': 0.045454545454545456,
     'adjust': 0.045454545454545456,
     'behave': 0.045454545454545456,
     'result': 0.09090909090909091,
     'sense': 0.045454545454545456,
     'develops': 0.045454545454545456,
     'able': 0.045454545454545456,
     'read': 0.045454545454545456,
     'situations': 0.045454545454545456,
     'determine': 0.045454545454545456,
     'pick': 0.045454545454545456,
     'worried': 0.09090909090909091,
     'concerns': 0.13636363636363635,
     'Famous': 0.045454545454545456,
     'scientist': 0.045454545454545456,
     'Sir': 0.09090909090909091,
     'Stephen': 0.09090909090909091,
     'Hawking': 0.09090909090909091,
     'spoke': 0.09090909090909091,
     'past': 0.045454545454545456,
     'far': 0.045454545454545456,
     'useful': 0.045454545454545456,
     'helpful': 0.045454545454545456,
     'smarter': 0.045454545454545456,
     'humans': 0.045454545454545456,
     'potentially': 0.045454545454545456,
     'cause': 0.09090909090909091,
     'problems': 0.09090909090909091,
     'future': 0.045454545454545456,
     'People': 0.045454545454545456,
     'expressed': 0.045454545454545456,
     'privacy': 0.045454545454545456,
     'critics': 0.09090909090909091,
     'online': 0.045454545454545456,
     'encourages': 0.045454545454545456,
     'concern': 0.045454545454545456,
     'intelligent': 0.045454545454545456,
     'usually': 0.045454545454545456,
     'leave': 0.045454545454545456,
     'unemployed': 0.045454545454545456,
     'disagree': 0.045454545454545456,
     'saying': 0.045454545454545456,
     'thoughts': 0.045454545454545456,
     'actions': 0.045454545454545456,
     'danger': 0.045454545454545456,
     'described': 0.045454545454545456,
     'thing': 0.09090909090909091,
     'bad': 0.045454545454545456,
     'Let': 0.045454545454545456}



#### Sentence Tokenisation


```python
sentence_token = [token for token in doc.sents]
sentence_token
```




    [
     What is Artifical intelligence?,
     (https://www.bbc.co.uk/newsround/49274918),
     Artificial intelligence - or AI for short - is technology that enables a computer to think or act in a more 'human' way.,
     It does this by taking in information from its surroundings, and deciding its response based on what it learns or senses.,
     It affects the the way we live, work and have fun in our spare time - and sometimes without us even realising.,
     AI is becoming a bigger part of our lives, as the technology behind it becomes more and more advanced.,
     Machines are improving their ability to 'learn' from mistakes and change how they approach a task the next time they try it.,
     Some researchers are even trying to teach robots about feelings and emotions.,
     You might not realise some of the devices and daily activities which rely on AI technology - phones, video games and going shopping, for example.,
     More technology,
     Why did this photo make history 60 years ago?,
     How robots and drones are changing deliveries,
     Flyboard inventor crosses English Channel,
     Why Instagram is going to hide your 'likes',
     Some people think that the technology is a really good idea, while others aren't so sure.,
     Just this month, it was announced that the NHS in England is setting up a special AI laboratory to boost the role of AI within the health service.,
     Announcing that the government will spend £250 million on this, Health Secretary Matt Hancock said the technology had "enormous power" to improve care, save lives and ensure doctors had more time to spend with patients.,
     Read on to find out more about AI and let us know what you think about it in the comments below.,
     What does AI do?,
     AI can be used for many different tasks and activities.,
     Personal electronic devices or accounts (like our phones or social media) use AI to learn more about us and the things that we like.,
     One example of this is entertainment services like Netflix which use the technology to understand what we like to watch and recommend other shows based on what they learn.,
     It can make video games more challenging by studying how a player behaves, while home assistants like Alexa and Siri also rely on it.,
     It has been announced that NHS England will spend millions on AI in order to improve patient care and research,
     AI can be used in healthcare, not only for research purposes, but also to take better care of patients through improved diagnosis and monitoring.,
     It also has uses within transport too.,
     For example, driverless cars are an example of AI tech in action, while it is used extensively in the aviation industry (for example, in flight simulators).,
     Farmers can use AI to monitor crops and conditions, and to make predictions, which will help them to be more efficient.,
     You only have to look at what some of these AI robots can do to see just how advanced the technology is and imagine many other jobs for which it could be used.,
     Where did AI come from?,
     The term 'artificial intelligence' was first used in 1956.,
     In the 1960s, scientists were teaching computers how to mimic - or copy - human decision-making.,
     This developed into research around 'machine learning', in which robots were taught to learn for themselves and remember their mistakes, instead of simply copying.,
     Algorithms play a big part in machine learning as they help computers and robots to know what to do.,
     What is an algorithm?,
     An algorithm is basically a set of rules or instructions which a computer can use to help solve a problem or come to a decision about what to do next.,
     From here, the research has continued to develop, with scientists now exploring 'machine perception'.,
     This involves giving machines and robots special sensors to help them to see, hear, feel and taste things like human do - and adjust how they behave as a result of what they sense.,
     The idea is that the more this technology develops, the more robots will be able to 'understand' and read situations, and determine their response as a result of the information that they pick up.,
     Why are people worried about AI?,
     Many people have concerns about AI technology and teaching robots too much.,
     Famous scientist Sir Stephen Hawking spoke out about it in the past.,
     He said that although the AI we've made so far has been very useful and helpful, he worried that if we teach robots too much, they could become smarter than humans and potentially cause problems.,
     Sir Stephen Hawking spoke out about AI and said that he had concerns that the technology could cause problems in the future,
     People have expressed concerns about privacy too.,
     For example, critics think that it could become a problem if AI learns too much about what we like to look at online and encourages us to spend too much time on electronic devices.,
     Another concern about AI is that if robots and computers become very intelligent, they could learn to do jobs which people would usually have to do, which could leave some people unemployed.,
     Other people disagree, saying that the technology will never be as advanced as human thoughts and actions, so there is not a danger of robots 'taking over' in the way that some critics have described.,
     What do you think about AI?,
     Do you think that it is a good thing or a bad thing?,
     Let us know in the comments below.]



#### Now we have a score of the sentences based on the score of the words to identify the importance of each sentence.


```python
sentence_score = {}
for sent in sentence_token:
    for word in sent:
        if word.text.lower() in word_frequencies:
                if sent not in sentence_score:
                    sentence_score[sent] = word_frequencies[word.text.lower()]
                else:
                    sentence_score[sent] += word_frequencies[word.text.lower()]

sentence_score
```




    {
     What is Artifical intelligence?: 0.13636363636363635,
     (https://www.bbc.co.uk/newsround/49274918): 0.045454545454545456,
     Artificial intelligence - or AI for short - is technology that enables a computer to think or act in a more 'human' way.: 1.5454545454545454,
     It does this by taking in information from its surroundings, and deciding its response based on what it learns or senses.: 0.5909090909090909,
     It affects the the way we live, work and have fun in our spare time - and sometimes without us even realising.: 0.5909090909090909,
     AI is becoming a bigger part of our lives, as the technology behind it becomes more and more advanced.: 0.8181818181818181,
     Machines are improving their ability to 'learn' from mistakes and change how they approach a task the next time they try it.: 0.8181818181818181,
     Some researchers are even trying to teach robots about feelings and emotions.: 0.7727272727272727,
     You might not realise some of the devices and daily activities which rely on AI technology - phones, video games and going shopping, for example.: 1.636363636363636,
     More technology: 0.5454545454545454,
     Why did this photo make history 60 years ago?: 0.2272727272727273,
     How robots and drones are changing deliveries: 0.6363636363636362,
     Flyboard inventor crosses English Channel: 0.09090909090909091,
     Why Instagram is going to hide your 'likes': 0.18181818181818182,
     Some people think that the technology is a really good idea, while others aren't so sure.: 1.318181818181818,
     Just this month, it was announced that the NHS in England is setting up a special AI laboratory to boost the role of AI within the health service.: 0.5,
     Announcing that the government will spend £250 million on this, Health Secretary Matt Hancock said the technology had "enormous power" to improve care, save lives and ensure doctors had more time to spend with patients.: 2.0909090909090904,
     Read on to find out more about AI and let us know what you think about it in the comments below.: 0.6363636363636364,
     AI can be used for many different tasks and activities.: 0.18181818181818182,
     Personal electronic devices or accounts (like our phones or social media) use AI to learn more about us and the things that we like.: 1.5909090909090906,
     One example of this is entertainment services like Netflix which use the technology to understand what we like to watch and recommend other shows based on what they learn.: 2.2727272727272725,
     It can make video games more challenging by studying how a player behaves, while home assistants like Alexa and Siri also rely on it.: 0.8636363636363638,
     It has been announced that NHS England will spend millions on AI in order to improve patient care and research: 0.8181818181818181,
     AI can be used in healthcare, not only for research purposes, but also to take better care of patients through improved diagnosis and monitoring.: 0.6818181818181818,
     It also has uses within transport too.: 0.09090909090909091,
     For example, driverless cars are an example of AI tech in action, while it is used extensively in the aviation industry (for example, in flight simulators).: 1.227272727272727,
     Farmers can use AI to monitor crops and conditions, and to make predictions, which will help them to be more efficient.: 0.5909090909090909,
     You only have to look at what some of these AI robots can do to see just how advanced the technology is and imagine many other jobs for which it could be used.: 1.409090909090909,
     Where did AI come from?: 0.09090909090909091,
     The term 'artificial intelligence' was first used in 1956.: 0.2727272727272727,
     In the 1960s, scientists were teaching computers how to mimic - or copy - human decision-making.: 0.7727272727272728,
     This developed into research around 'machine learning', in which robots were taught to learn for themselves and remember their mistakes, instead of simply copying.: 1.4999999999999998,
     Algorithms play a big part in machine learning as they help computers and robots to know what to do.: 1.2727272727272725,
     What is an algorithm?: 0.09090909090909091,
     An algorithm is basically a set of rules or instructions which a computer can use to help solve a problem or come to a decision about what to do next.: 1.0454545454545454,
     From here, the research has continued to develop, with scientists now exploring 'machine perception'.: 0.5909090909090908,
     This involves giving machines and robots special sensors to help them to see, hear, feel and taste things like human do - and adjust how they behave as a result of what they sense.: 1.9090909090909087,
     The idea is that the more this technology develops, the more robots will be able to 'understand' and read situations, and determine their response as a result of the information that they pick up.: 1.772727272727272,
     Why are people worried about AI?: 0.36363636363636365,
     Many people have concerns about AI technology and teaching robots too much.: 1.5454545454545454,
     Famous scientist Sir Stephen Hawking spoke out about it in the past.: 0.18181818181818182,
     He said that although the AI we've made so far has been very useful and helpful, he worried that if we teach robots too much, they could become smarter than humans and potentially cause problems.: 1.2727272727272725,
     Sir Stephen Hawking spoke out about AI and said that he had concerns that the technology could cause problems in the future: 1.1363636363636362,
     People have expressed concerns about privacy too.: 0.5,
     For example, critics think that it could become a problem if AI learns too much about what we like to look at online and encourages us to spend too much time on electronic devices.: 1.9090909090909092,
     Another concern about AI is that if robots and computers become very intelligent, they could learn to do jobs which people would usually have to do, which could leave some people unemployed.: 1.727272727272727,
     Other people disagree, saying that the technology will never be as advanced as human thoughts and actions, so there is not a danger of robots 'taking over' in the way that some critics have described.: 2.227272727272727,
     What do you think about AI?: 0.2727272727272727,
     Do you think that it is a good thing or a bad thing?: 0.5909090909090909,
     Let us know in the comments below.: 0.2727272727272727}



#### The next goal is to reduce the summary from 100% down to 20% with high scoring sentences


```python
from heapq import nlargest
```


```python
select_length = int(len(sentence_token) * 0.2)
print("Maximum number of sentences needed to achieve a 80% reducion:",select_length)
```

    Maximum number of sentences needed to achieve a 80% reducion: 10
    


```python
summary = nlargest(select_length, sentence_score, key=sentence_score.get)
```

#### Finally, we have the list of important sentences so all that is required is to put it all together


```python
summary
```




    [One example of this is entertainment services like Netflix which use the technology to understand what we like to watch and recommend other shows based on what they learn.,
     Other people disagree, saying that the technology will never be as advanced as human thoughts and actions, so there is not a danger of robots 'taking over' in the way that some critics have described.,
     Announcing that the government will spend £250 million on this, Health Secretary Matt Hancock said the technology had "enormous power" to improve care, save lives and ensure doctors had more time to spend with patients.,
     For example, critics think that it could become a problem if AI learns too much about what we like to look at online and encourages us to spend too much time on electronic devices.,
     This involves giving machines and robots special sensors to help them to see, hear, feel and taste things like human do - and adjust how they behave as a result of what they sense.,
     The idea is that the more this technology develops, the more robots will be able to 'understand' and read situations, and determine their response as a result of the information that they pick up.,
     Another concern about AI is that if robots and computers become very intelligent, they could learn to do jobs which people would usually have to do, which could leave some people unemployed.,
     You might not realise some of the devices and daily activities which rely on AI technology - phones, video games and going shopping, for example.,
     Personal electronic devices or accounts (like our phones or social media) use AI to learn more about us and the things that we like.,
     Artificial intelligence - or AI for short - is technology that enables a computer to think or act in a more 'human' way.]




```python
final_summary = [word.text for word in summary]
summary = "".join(final_summary)
```


```python
print(summary)
```

    One example of this is entertainment services like Netflix which use the technology to understand what we like to watch and recommend other shows based on what they learn.
    Other people disagree, saying that the technology will never be as advanced as human thoughts and actions, so there is not a danger of robots 'taking over' in the way that some critics have described.
    Announcing that the government will spend £250 million on this, Health Secretary Matt Hancock said the technology had "enormous power" to improve care, save lives and ensure doctors had more time to spend with patients.
    For example, critics think that it could become a problem if AI learns too much about what we like to look at online and encourages us to spend too much time on electronic devices.
    This involves giving machines and robots special sensors to help them to see, hear, feel and taste things like human do - and adjust how they behave as a result of what they sense.
    The idea is that the more this technology develops, the more robots will be able to 'understand' and read situations, and determine their response as a result of the information that they pick up.
    Another concern about AI is that if robots and computers become very intelligent, they could learn to do jobs which people would usually have to do, which could leave some people unemployed.
    You might not realise some of the devices and daily activities which rely on AI technology - phones, video games and going shopping, for example.
    Personal electronic devices or accounts (like our phones or social media) use AI to learn more about us and the things that we like.Artificial intelligence - or AI for short - is technology that enables a computer to think or act in a more 'human' way.
    


```python
print("Length of original text:", len(text))
print("Length of summary:", len(summary))
```

    Length of original text: 5210
    Length of summary: 1741
    
