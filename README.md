---
# Ähnlichkeit von Job-Stellenausschreibungen mit Word2Vec-Ansatz
by Tobia Ippolito

---

## Table of Content

- [Aufgabenstellung](#Aufgabenstellung)
- [Wann ist eine Stellenausschreibung ähnlich?](#Wann-ist-eine-Stellenausschreibung-ähnlich?)
- [Grundgedanke](#Grundgedanke)
- [Imports](#Imports)
- [Preparations](#Preparation)
- Experimente
    - [Word2Vec Similarity der Jobbeschreibung](#Experiment-1:-Word2Vec-Similarity-der-Jobbeschreibung)
    - [Word2Vec der Jobbeschreibung mit nur wichtigen Wörtern](#Experiment-2:-Word2Vec-der-Jobbeschreibung-mit-nur-wichtigen-Wörtern)
    - [Word2Vec der Jobbeschreibung mit nur Entitäten](#Experiment-3:-Word2Vec-der-Jobbeschreibung-mit-nur-Entitäten)
    - [Word2Vec der Jobbeschreibung mit nur wichtigen Wörtern und anderer Ausrechnungsart 1](#Experiment-4:-Word2Vec-der-Jobbeschreibung-mit-nur-wichtigen-Wörtern-und-anderer-Ausrechnungsart-1)
    - [Word2Vec der Jobbeschreibung mit nur wichtigen Wörtern und anderer Ausrechnungsart 2](#Experiment-5:-Word2Vec-der-Jobbeschreibung-mit-nur-wichtigen-Wörtern-und-anderer-Ausrechnungsart-2)
    - [Ähnlichkeit der Job-Titel mit Word2vec](#Experiment-6:-Ähnlichkeit-der-Job-Titel-mit-Word2vec)
    - [Ähnlichkeit der Job-Kategorien](#Experiment-7:-Ähnlichkeit-der-Job-Kategorien)
    - [Ähnlichkeit der Job-Orte](#Experiment-8:-Ähnlichkeit-der-Job-Orte)
    - [Ähnlichkeit der Job-Art](#Experiment-9:-Ähnlichkeit-der-Job-Art)
    - [Point-Gain-Proceedings](#Experiment-10:-Point-Gain-Proceedings)
    - [Parallelisierung](#Experiment-11:-Paralleles-Ausführen)
- [Evaluation](#Evaluation)
- [Anwendung](#Anwendung)

<br>

> ***Hinweis**: Klicken Sie sich einfach zu den jeweiligen Kapiteln. Dort befinden sich rechts Racketen, welche sie mit einem Klick wieder hier hin bringen, wo Sie von neuem weiter navigieren können.* <img src="./rackete_1.png" style="float:right" width=50></img>

<br>

---

### Aufgabenstellung

Finden von ähnlichen Job-Posts
Stellen Sie sich vor, ein/e Bewerber/in hat eine für ihn / sie interessante Stelle gefunden. Nun
würde er/sie gerne automatisch ähnliche Stellen vorgeschlagen bekommen, die auch
interessant sein könnten. Wie könnte eine entsprechende Funktionalität implementiert
werden?

> Definieren Sie zunächst, wann eine Stelle in diesem Szenario als ähnlich gesehen
werden sollte.

> Überlegen Sie dann, mit welchen Ansätzen diese Art von Ähnlichkeit erfasst werden kann.

> Setzen Sie diese anschließend um. Verwenden Sie Experimente und gehen Sie strukturiert vor.

<br>

---

### Wann ist eine Stellenausschreibung ähnlich?

Eigenschaften eine Job-Posts:
- Job Beschreibung
- Job Titel
- Kategorie/Thema
- Ort (Stadt, Land, Unternehmen)
- Plattform
- Berufsart


Arten der Ähnlichkeit bei Job-Posts:
- Ansprüche 
- Fachgebiet
- Spezieller Beruf/Bezeichnung (Ähnlichkeit der Aktivität)
- Gehalt
- Firma
- Plattform
- Ort
- Berufsart (Teilszeit, ...)
- Textlänge
- Grammatik


Tatsächlich ist keine objektive Ähnlichkeit festzulegen. Stattdessen ist es subjektiv, wie ähnlich sich Berufstellenanzeigen sind. So ist für eine Person ein naher/ähnlicher Ort wichtig und für eine andere, dass es Vollzeit ist.<br>
Generell gehe ich jedoch davon aus, dass die wichtigste Eigenschaft der Beruf an sich ist (Berufsbezeichnung).

---
### Grundgedanke

Zum einen soll die Suche flexibel und individuell anpassbar sein, da wir [gerade](#Wann-ist-eine-Stellenausschreibung-ähnlich?) gesehen haben, dass die Ähnlichkeit bei Job-Posts subjektiv ist.<br>
Zum anderen soll das Verfahren Word2Vec zur verwendung kommen. Dieses Verfahren stellt Wörter nämlich so dar, dass ähnliche Wörter mit repräsentiert werden. Dies wollen wir uns in dieser Arbeit zunutze machen.<br>
Ein Job-Post besteht zu einem großen Teil aus Wörtern, welche einen semantischen Sinn haben und welcher auch von großer Bedeutung ist -> Jobbezeichnung + Jobbeschreibung.

---

### Imports


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
import seaborn as sn

import spacy
from spacy.tokens import Token

from geopy.geocoders import Nominatim
from geopy.distance import geodesic

import multiprocessing as mp

import dask as d
import dask.dataframe as dd
from dask.delayed import delayed
```

---

### Preparation
<!--<p style="float:right">Up</p>-->

[<img src="./rackete_1.png" style="float:right" width=100></img>](#Table-of-Content)

**Loading Data**


```python
!pip install openpyxl
```

    Requirement already satisfied: openpyxl in c:\users\tobia\anaconda3\envs\ai\lib\site-packages (3.0.10)
    Requirement already satisfied: et-xmlfile in c:\users\tobia\anaconda3\envs\ai\lib\site-packages (from openpyxl) (1.1.0)



```python
data = pd.read_excel("../data_scientist_united_states_job_postings_jobspikr.xlsx")
```


```python
data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>crawl_timestamp</th>
      <th>url</th>
      <th>job_title</th>
      <th>category</th>
      <th>company_name</th>
      <th>city</th>
      <th>state</th>
      <th>country</th>
      <th>inferred_city</th>
      <th>inferred_state</th>
      <th>...</th>
      <th>job_description</th>
      <th>job_type</th>
      <th>salary_offered</th>
      <th>job_board</th>
      <th>geo</th>
      <th>cursor</th>
      <th>contact_email</th>
      <th>contact_phone_number</th>
      <th>uniq_id</th>
      <th>html_job_description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2019-02-06 06:26:22</td>
      <td>https://www.indeed.com/viewjob?jk=fd83355c2b23...</td>
      <td>Enterprise Data Scientist I</td>
      <td>Accounting/Finance</td>
      <td>Farmers Insurance Group</td>
      <td>Woodland Hills</td>
      <td>CA</td>
      <td>Usa</td>
      <td>Woodland hills</td>
      <td>California</td>
      <td>...</td>
      <td>Read what people are saying about working here...</td>
      <td>Undefined</td>
      <td>NaN</td>
      <td>indeed</td>
      <td>usa</td>
      <td>1549432819114777</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>3b6c6acfcba6135a31c83bd7ea493b18</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2019-02-06 06:33:41</td>
      <td>https://www.dice.com/jobs/detail/Data-Scientis...</td>
      <td>Data Scientist</td>
      <td>NaN</td>
      <td>Luxoft USA Inc</td>
      <td>Middletown</td>
      <td>NJ</td>
      <td>Usa</td>
      <td>Middletown</td>
      <td>New jersey</td>
      <td>...</td>
      <td>We have an immediate opening for a Sharp Data ...</td>
      <td>Undefined</td>
      <td>NaN</td>
      <td>dice</td>
      <td>usa</td>
      <td>1549432819122106</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>741727428839ae7ada852eebef29b0fe</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2019-02-06 06:33:35</td>
      <td>https://www.dice.com/jobs/detail/Data-Scientis...</td>
      <td>Data Scientist</td>
      <td>NaN</td>
      <td>Cincinnati Bell Technology Solutions</td>
      <td>New York</td>
      <td>NY</td>
      <td>Usa</td>
      <td>New york</td>
      <td>New york</td>
      <td>...</td>
      <td>Candidates should have the following backgroun...</td>
      <td>Full Time</td>
      <td>NaN</td>
      <td>dice</td>
      <td>usa</td>
      <td>1549432819236156</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>cdc9ef9a1de327ccdc19cc0d07dbbb37</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2019-02-06 06:33:42</td>
      <td>https://www.indeed.com/viewjob?jk=841edd86ead2...</td>
      <td>Data Scientist, Aladdin Wealth Tech, Associate...</td>
      <td>Accounting/Finance</td>
      <td>BlackRock</td>
      <td>New York</td>
      <td>NY 10055 (Midtown area)</td>
      <td>Usa</td>
      <td>New york</td>
      <td>New york</td>
      <td>...</td>
      <td>Read what people are saying about working here...</td>
      <td>Undefined</td>
      <td>NaN</td>
      <td>indeed</td>
      <td>usa</td>
      <td>1549432819259473</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1c8541cd2c2c924f9391c7d3f526f64e</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2019-02-06 06:48:23</td>
      <td>https://job-openings.monster.com/senior-data-s...</td>
      <td>Senior Data Scientist</td>
      <td>biotech</td>
      <td>CyberCoders</td>
      <td>Charlotte</td>
      <td>NC</td>
      <td>Usa</td>
      <td>Charlotte</td>
      <td>North carolina</td>
      <td>...</td>
      <td>We are seeking an extraordinary Data Scientist...</td>
      <td>Full Time</td>
      <td>NaN</td>
      <td>monster</td>
      <td>usa</td>
      <td>1549436429015957</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>445652a560a5441060857853cf267470</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 22 columns</p>
</div>




```python
data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 10000 entries, 0 to 9999
    Data columns (total 22 columns):
     #   Column                Non-Null Count  Dtype         
    ---  ------                --------------  -----         
     0   crawl_timestamp       10000 non-null  datetime64[ns]
     1   url                   10000 non-null  object        
     2   job_title             10000 non-null  object        
     3   category              9118 non-null   object        
     4   company_name          9999 non-null   object        
     5   city                  9751 non-null   object        
     6   state                 9584 non-null   object        
     7   country               10000 non-null  object        
     8   inferred_city         8980 non-null   object        
     9   inferred_state        9484 non-null   object        
     10  inferred_country      9505 non-null   object        
     11  post_date             10000 non-null  datetime64[ns]
     12  job_description       10000 non-null  object        
     13  job_type              10000 non-null  object        
     14  salary_offered        597 non-null    object        
     15  job_board             9310 non-null   object        
     16  geo                   9300 non-null   object        
     17  cursor                10000 non-null  int64         
     18  contact_email         0 non-null      float64       
     19  contact_phone_number  416 non-null    object        
     20  uniq_id               10000 non-null  object        
     21  html_job_description  1599 non-null   object        
    dtypes: datetime64[ns](2), float64(1), int64(1), object(18)
    memory usage: 1.7+ MB



```python
data.shape
```




    (10000, 22)



---

Download bigger SpaCy train text with:<br>
python -m spacy download en_core_web_lg

Or in Anaconda:<br>
conda install spacy-model-en_core_web_lg

Create an Example:

- job1 = Farmer job    - Accounting/Finance
- job2 = Finance job (investing, black rock)   - Accounting/Finance
- job3 = Data Science (analytics)
- job4 = Data Science (small text, machine learning)
- job5 = Data Science 


```python
job1 = data['job_description'][0]    # farmer
job2 = data['job_description'][3]    # finance -> inevsting, black rock
job3 = data['job_description'][4]    # data science -> analytics / bio
job4 = data['job_description'][2]    # data science -> machine learning
job5 = data['job_description'][1]    # data science -> mathematics

jobs_ = (job1, job2, job3, job4, job5)
```

Evaluation Function for evaulate a result


```python
def eval_similarity(similarity_matrix, labels, heatmap=True, k=1):
    """
    Evaluation function for similarity.
    """
    if not heatmap:
        fig, ax = plt.subplots(nrows=(len(labels)+1)//2, ncols=2, figsize=(15, 15))  #, constrained_layout=True
        #fig.tight_layout()
        plt.subplots_adjust(hspace = 0.3)

        row_ = 0
        col_ = 0

        for i, sim in enumerate(similarity_matrix):
            # create cmap with normalize values
            my_cmap = plt.cm.get_cmap('GnBu')
            colors = my_cmap([x / max(sim) for x in sim])
            # plot one job simiralities
            ax[row_][col_].set_title(f"Similarities from job{i+1}")
            ax[row_][col_].bar(labels, sim, edgecolor='black', color=colors)
            ax[row_][col_].grid()

            # add colorbar
            sm = ScalarMappable(cmap=my_cmap, norm=plt.Normalize(0,max(sim)))
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=ax[row_][col_])
            cbar.set_label('Color', rotation=270,labelpad=25)

            # update col and row for axes indices
            if col_ == 1:
                col_ = 0
                row_ += 1
            else:
                col_ += 1

        plt.plot()
    else:
        # create mask for only half
        mask = np.triu(similarity_matrix, k=k)
        # np.ones_like(similarity_matrix, dtype=bool)
        #print(mask)
        
        df_color_map = pd.DataFrame(similarity_matrix, index=labels, columns=labels)
        plt.figure(figsize = (10,7))
        plt.title("Job Similarities")
        sn.heatmap(df_color_map, annot=True, cmap=sn.cm.rocket_r, mask=mask)
```


```python
def create_similarity_matrix(sim_func, jobs):
    similarity_matrix = []
    for i in jobs:
        cache = []
        for j in jobs:
            cache += [sim_func(i, j)]
        similarity_matrix += [cache]
    return similarity_matrix
```

**Hinweis an den Leser:**

Bei der Evaluierung der folgenden Experimente wäre folgendes Ergebnis ein optimales Ergebnis. Nur um dies besser einordnen zu können.


```python
template = [[1, 0.7, 0.3, 0.3, 0.3], 
           [0.7, 1, 0.3, 0.3, 0.3],
           [0.3, 0.3, 1, 0.95, 0.95],
           [0.3, 0.3, 0.95, 1, 0.95],
           [0.3, 0.3, 0.95, 0.95, 1]]
eval_similarity(template, ["job1", "job2", "job3", "job4", "job5"])
```


​    
![png](res/word2vec-job-posts-similarity_23_0.png)
​    


Die Jobs 3-5 sind ähnliche Jobs und sonst sind alle verschieden.

---


### **Experiment 1:** Word2Vec Similarity der Jobbeschreibung

In diesem Experiment soll die Ähnlichkeit der Stellenausschreibungen mithilfe des Word2Vec-Distanz der Job-Beschreibungen evaluiert werden.<br>
Die Jobbeschreibung wird in diesem Experiment nicht verändert.

[<img src="./rackete_1.png" style="float:right" width=100></img>](#Table-of-Content)


```python
nlp = spacy.load("en_core_web_lg")

doc1 = nlp(job1)
doc2 = nlp(job2)
doc3 = nlp(job3)
doc4 = nlp(job4)
doc5 = nlp(job5)
```


```python
jobs = (doc1, doc2, doc3, doc4, doc5)
job_labels = ["job1", "job2", "job3", "job4", "job5"]
```


```python
nlp(job1)[:].similarity(nlp(job2)[:])
```




    0.9840876460075378




```python
nlp(job1)[:].similarity(nlp(job3)[:])
```




    0.9855178594589233




```python
nlp(job2)[:].similarity(nlp(job3)[:])
```




    0.9882321357727051




```python
nlp(job1)[:].similarity(nlp(job5)[:])
```




    0.9802072644233704




```python
nlp(job3)[:].similarity(nlp(job4)[:])
```




    0.9440867304801941




```python
nlp(job3)[:].similarity(nlp(job5)[:])
```




    0.9785909652709961




```python
nlp(job4)[:].similarity(nlp(job5)[:])
```




    0.9545732736587524




```python
similarity = create_similarity_matrix(lambda x,y: x.similarity(y), jobs)
```


```python
eval_similarity(similarity, job_labels, heatmap=True)
```


​    
![png](res/word2vec-job-posts-similarity_38_0.png)
​    


**Ergebnis:**

Zunächst fällt auf, dass alle Jobbeschreibungen als sehr ähnlich eingestuft werden, was diesen Ansatz schonmal für nicht so optimal einstuft.<br> 
Nur die 4.te Jobbeschreibung scheint nicht so ähnlich wie die 4 anderen Jobbeschreibungen zu sein. Die Jobbeschreibung dessen ist auch eindeutig kürzer und es könnte sich damit um einen Ausreißer handeln.<br>
<br>
So oder so ist dieser Ansatz nicht zu gebrauchen, da die Jobbeschreibungen sich damit nicht trennen lassen.

---
### **Experiment 2:** Word2Vec der Jobbeschreibung mit nur wichtigen Wörtern

In diesem Experiment soll die Ähnlichkeit der Stellenausschreibungen mithilfe des Word2Vec-Distanz der Job-Beschreibungen evaluiert werden.<br>
Die Jobbeschreibung wird in diesem Experiment auf die wichtigsten Wörter reduziert (Nomen, Adjektiven). Die Auswirkung auf die Ähnlichkeit wird hierbei getestet.

[<img src="./rackete_1.png" style="float:right" width=100></img>](#Table-of-Content)


```python
# remove stopwords and unimportant words in Spacy Pipeline
def get_important_words(doc):
    """
    Prepares the job description and removes all words except nouns and verbs
    """
    cache = []
    for token in doc:
        if token.pos_ in ['NOUN', 'PROPN', 'ADJ']:    #'VERB',
            cache += [token.text]
    return cache
```


```python
#[print(i.text, i.pos_) for i in nlp(job3)]
```


```python
# remove nouns and than build a new doc out of these smaller descriptions

nlp = spacy.load("en_core_web_lg")

# remove not important words
prepared_jobs = []
for job in jobs_:
    prepared_jobs += [get_important_words(nlp(job))]
    
# build docs with the reduced text

doc1 = nlp(' '.join(prepared_jobs[0]))
doc2 = nlp(' '.join(prepared_jobs[1]))
doc3 = nlp(' '.join(prepared_jobs[2]))
doc4 = nlp(' '.join(prepared_jobs[3]))
doc5 = nlp(' '.join(prepared_jobs[4]))

jobs = (doc1, doc2, doc3, doc4, doc5)
job_labels = ["job1", "job2", "job3", "job4", "job5"]
```


```python
sim_matrix = create_similarity_matrix(lambda x,y: x.similarity(y), jobs)
```


```python
eval_similarity(sim_matrix, job_labels)
```


​    
![png](res/word2vec-job-posts-similarity_46_0.png)
​    


**Ergebnis:**

Die Ähnlichkeiten sind sich ziemlich ähnlich, nur die Jobbeschreibung 4 ist zu allen anderen ca. 5-10% unähnlicher.
Die Ergebnisse sind somit fast identisch zu denen des Experiments 1. Einziger Unterschied ist, dass die Werte etwas unterhalb liegen.<br>
<br>
Auch dieser Ansatz ist unbrauchbar, da die Ähnlichkeiten nicht mit der Realität übereinstimmen.

---

### **Experiment 3:** Word2Vec der Jobbeschreibung mit nur Entitäten

In diesem Experiment soll die Ähnlichkeit der Stellenausschreibungen mithilfe des Word2Vec-Distanz der Job-Beschreibungen evaluiert werden.<br>
Die Jobbeschreibung wird in diesem Experiment auf die Entitäten reduziert. Die Auswirkung auf die Ähnlichkeit wird hierbei getestet.

[<img src="./rackete_1.png" style="float:right" width=100></img>](#Table-of-Content)


```python
# build nlp of ents

nlp = spacy.load("en_core_web_lg")

# get only entities
ents_ = []
for job in jobs_:
    cache = ""
    for ent in nlp(job).ents:
        cache += f"{ent.text} " 
    ents_ += [cache]
    
# wrap in doc for similarity calculation
jobs = [nlp(ent) for ent in ents_]
job_labels = ["job1", "job2", "job3", "job4", "job5"]
```


```python
similarity = create_similarity_matrix(lambda x,y: x.similarity(y), jobs)
```


```python
eval_similarity(similarity, job_labels, heatmap=True)
```


​    
![png](res/word2vec-job-posts-similarity_52_0.png)
​    


**Ergebnis:**

Der Ansatz erzielt keine guten Resultate. Job 2 ist hiernach ähnlich zu den jobs 3 und 5. Genauso verhält es sich bei dem Job 1 und zwar noch extremer und gerade diese sind ja nicht ähnlich. Die Jobs 3-5 werden hierbei auch nur als etwas ähnlich eingestuft.

---
### **Experiment 4:** Word2Vec der Jobbeschreibung mit nur wichtigen Wörtern und anderer Ausrechnungsart 1

In diesem Experiment soll die Ähnlichkeit der Stellenausschreibungen mithilfe des Word2Vec-Distanz der Job-Beschreibungen evaluiert werden.<br>
Dabei wird die Ähnlichkeit so berechnet, dass die Ähnlichkeit von jedem Wort zu jedem Wort summiert wird.

[<img src="./rackete_1.png" style="float:right" width=100></img>](#Table-of-Content)


```python
# remove nouns and than build a new doc out of these smaller descriptions
nlp = spacy.load("en_core_web_lg")

# remove not important words
prepared_jobs = []
for job in jobs_:
    prepared_jobs += [get_important_words(nlp(job))]
    
# build docs with the reduced text

doc1 = nlp(' '.join(prepared_jobs[0]))
doc2 = nlp(' '.join(prepared_jobs[1]))
doc3 = nlp(' '.join(prepared_jobs[2]))
doc4 = nlp(' '.join(prepared_jobs[3]))
doc5 = nlp(' '.join(prepared_jobs[4]))

jobs = (doc1, doc2, doc3, doc4, doc5)
job_labels = ["job1", "job2", "job3", "job4", "job5"]
```


```python
# sum of similarities
def job_description_sum_similarity(post1, post2):
    """
    Calculates the similarity between 2 job descriptions.
    Both descriptions should be cleaned (stopword removing)
    and given as list/doc.
    """
    text_similarity = dict()
    # calculate similarity sum
    for token in post1:
        sum_sim = 0
        for token_other in post2:
            sum_sim += token.similarity(token_other)

        # saves similarity sum in dict
        label = token.text
        i = 1
        while label in text_similarity.keys():
            label = f"token.text{i}"
            i += 1
        text_similarity[label] = sum_sim
        
    return sum(text_similarity.values())/((len(post1)+len(post2))*100)
```


```python
sim_matrix = create_similarity_matrix(job_description_sum_similarity, jobs)
```

    C:\Users\tobia\AppData\Local\Temp\ipykernel_22936\869645819.py:13: UserWarning: [W008] Evaluating Token.similarity based on empty vectors.
      sum_sim += token.similarity(token_other)



```python
eval_similarity(sim_matrix, job_labels)
```


​    
![png](res/word2vec-job-posts-similarity_59_0.png)
​    


**Ergebnis:** 

Dieses Verfahren zeigt bei den gleichen Posts eine Unähnlichkeit an, womit es damit eindeutig unbrauchbar ist und Ähnlichkeit nicht gut berechnen kann. <br>
Auch bei den restlichen Resultate (wie damit dann auch zu erwarten ist) schneidet das verfahren sehr schlecht ab. So Sind die Jobbeschreibungen 1-3 fast gleich ähnlich zu der Jobbeschreibung 5, was in der Realität nicht der Fall ist.

---
### **Experiment 5:** Word2Vec der Jobbeschreibung mit nur wichtigen Wörtern und anderer Ausrechnungsart 2

In diesem Experiment soll die Ähnlichkeit der Stellenausschreibungen mithilfe des Word2Vec-Distanz der Job-Beschreibungen evaluiert werden.<br>
Dabei wird die Ähnlichkeit so berechnet, dass ähnliche Wörter gezählt werden.

[<img src="./rackete_1.png" style="float:right" width=100></img>](#Table-of-Content)


```python
# remove nouns and than build a new doc out of these smaller descriptions
nlp = spacy.load("en_core_web_lg")

# remove not important words
prepared_jobs = []
for job in jobs_:
    prepared_jobs += [get_important_words(nlp(job))]
    
# build docs with the reduced text

doc1 = nlp(' '.join(prepared_jobs[0]))
doc2 = nlp(' '.join(prepared_jobs[1]))
doc3 = nlp(' '.join(prepared_jobs[2]))
doc4 = nlp(' '.join(prepared_jobs[3]))
doc5 = nlp(' '.join(prepared_jobs[4]))

jobs = (doc1, doc2, doc3, doc4, doc5)
job_labels = ["job1", "job2", "job3", "job4", "job5"]
```


```python
# Or Count similar words
# go throw tokens and calc similarity
    # full similarity -> sum with all other tokens
def job_description_similarity_counter(post1, post2, sim_lim=0.75):
    """
    Calculates the similarity between 2 job descriptions with counters.
    Both descriptions should be cleaned (stopword removing)
    and given as list/doc.
    Returns a dictionary and the sum of the counts
    """
    text_similarity = dict()
    # calculate similarity counts
    for token in post1:
        counter_sim = 0
        for token_other in post2:
            if token.similarity(token_other) >= sim_lim:
                counter_sim += 1

        # save similarity counts in dict
        label = token.text
        i = 1
        while label in text_similarity.keys():
            label = f"token.text{i}"
            i += 1
        text_similarity[label] = counter_sim
        
    return sum(text_similarity.values())/(len(post1)+len(post2))
```


```python
sim_matrix = create_similarity_matrix(job_description_similarity_counter, jobs)
```

    C:\Users\tobia\AppData\Local\Temp\ipykernel_22936\4255546514.py:16: UserWarning: [W008] Evaluating Token.similarity based on empty vectors.
      if token.similarity(token_other) >= sim_lim:



```python
eval_similarity(sim_matrix, job_labels)
```


​    
![png](res/word2vec-job-posts-similarity_66_0.png)
​    


**Ergebnis:**

Die Ähnlichkeiten mit diesem Verfahren sind nicht wie gewünscht. Generell sind die Ähnlichkeiten sehr gering.<br>
So werden Job 3 und Job 1 als ähnlich deklariert und die Jobs 3-5 nicht.

---

### **Experiment 6:** Ähnlichkeit der Job-Titel mit Word2vec

In diesem Experiment soll die Ähnlichkeit der Stellenausschreibungen mithilfe des Word2Vec-Distanz der Job-Titeln evaluiert werden. 

[<img src="./rackete_1.png" style="float:right" width=100></img>](#Table-of-Content)

Load the title of these jobs


```python
job1_title = data['job_title'][0]    # farmer
job2_title = data['job_title'][3]    # finance -> inevsting, black rock
job3_title = data['job_title'][4]    # data science -> analytics
job4_title = data['job_title'][2]    # data science -> machine learning
job5_title = data['job_title'][1]

titles_ = [job1_title, job2_title, job3_title, job4_title, job5_title]
```

Take in SpaCy Pipeline to get Word2Vec value


```python
nlp = spacy.load("en_core_web_lg")

doc_titles = [nlp(title) for title in titles_]
labels = ["job1", "job2", "job3", "job4", "job5"]
```

Evaluate


```python
sim_matrix = create_similarity_matrix(lambda x,y: x.similarity(y), doc_titles)
```


```python
eval_similarity(sim_matrix, labels)
```


​    
![png](res/word2vec-job-posts-similarity_76_0.png)
​    


**Ergebnis:**

Das Verfahren zeigt recht gute Ergebnisse. Die Jobs 3-5 werden als ähnlich gekennzeichnet. Währendessen ist die Ähnlichkeit zu dem 2.ten Job der Jobs 3-5 merkbar geringer. Ebenso wird die Unähnlichkeit der Jobs 1 und 2 erkannt.<br>
<br>
Die einzige Schwäche ist die Ähnlichkeit von Job 1 und den jobs 3-5. Die Ähnlichkeit ist hier zwar etwas geinger, jedoch nicht so stark.

---

### **Experiment 7:** Ähnlichkeit der Job-Kategorien

In diesem Experiment soll die Ähnlichkeit der Stellenausschreibungen mithilfe der Kategorie des Jobs berechnet werden.

[<img src="./rackete_1.png" style="float:right" width=100></img>](#Table-of-Content)

Load the city of these jobs


```python
job1_category = data['category'][0]    # farmer
job2_category = data['category'][3]    # finance -> inevsting, black rock
job3_category = data['category'][4]    # data science -> analytics
job4_category = data['category'][2]    # data science -> machine learning
job5_category = data['category'][1]

categories_ = [job1_category, job2_category, job3_category, job4_category, job5_category]
```


```python
def get_most_common_noun(job_description):
    nlp = spacy.load("en_core_web_lg")
    doc = nlp(job_description)
    words = dict()
    for token in doc:
        if token.pos_ in ["NOUN", "PROPN"]:
            if token.text in words.keys():
                words[token.text] += 1
            else:
                words[token.text] = 1
    return sorted(words.items(), key=lambda x:x[0])[0][0]           
```


```python
get_most_common_noun(jobs_[4])
```




    'Algorithms'




```python
get_most_common_noun(jobs_[1])
```




    '@blackrock'




```python
# if no category -> find category
# get most common noun
fixed_categories_ = []
for i, category in enumerate(categories_):
    if type(category) == float:
        # overwrite it
        fixed_categories_ += [get_most_common_noun(jobs_[i])]
    else:
        fixed_categories_ += [category]
```

Take in SpaCy Pipeline to get Word2Vec value


```python
nlp = spacy.load("en_core_web_lg")

doc_categories = [nlp(category) for category in fixed_categories_]
labels = ["job1", "job2", "job3", "job4", "job5"]
```


```python
fixed_categories_
```




    ['Accounting/Finance', 'Accounting/Finance', 'biotech', 'Apache', 'Algorithms']



Evaluate


```python
sim_matrix = create_similarity_matrix(lambda x,y: x.similarity(y), doc_categories)
```


```python
eval_similarity(sim_matrix, labels)
```


​    
![png](res/word2vec-job-posts-similarity_91_0.png)
​    


**Ergebnis:**

Das Verfahren scheint nicht ganz zu funktionieren. Die Jobs 1 & 2 wurden als ähnlich deklariert, was gut ist. Die 3 Data Science Berufe jedoch gar nicht. Damit ist dieses Verfahren so erstmal nur zum Teil hilfreich.<br>
<br>
Es könnte so implementiert werden, dass bei ähnlicher Job-Kategorie Bonuspunkte vergeben werden (siehe letztes Experiment). Es muss jedoch bedacht werden, dass viele ähnliche Job-Kategorien nicht gefunden werden! Hier sind die Bezeichnungen zu unterschiedlich.

---

### **Experiment 8:** Ähnlichkeit der Job-Orte

In diesem Experiment soll die Ähnlichkeit der Stellenausschreibungen mithilfe der Ortsdistanz berechnet werden.

[<img src="./rackete_1.png" style="float:right" width=100></img>](#Table-of-Content)

Load the city of these jobs


```python
job1_city = data['city'][0]    # farmer
job2_city = data['city'][3]    # finance -> inevsting, black rock
job3_city = data['city'][4]    # data science -> analytics
job4_city = data['city'][2]    # data science -> machine learning
job5_city = data['city'][1]

cities_ = [job1_city, job2_city, job3_city, job4_city, job5_city]
labels = ["job1", "job2", "job3", "job4", "job5"]
```


```python
job1_country = data['country'][0]    # farmer
job2_country = data['country'][3]    # finance -> inevsting, black rock
job3_country = data['country'][4]    # data science -> analytics
job4_country = data['country'][2]    # data science -> machine learning
job5_country = data['country'][1]

countries_ = [job1_country, job2_country, job3_country, job4_country, job5_country]
```


```python
def find_pos(city, country):
    # Initialize Nominatim API
    geolocator = Nominatim(user_agent="MyApp")

    location = geolocator.geocode(f"{city} {country}", language="en")
    return location.latitude, location.longitude
```


```python
find_pos("paris", "france")
```




    (48.8588897, 2.3200410217200766)




```python
positions = []
for job in range(5):
    positions += [find_pos(cities_[job], countries_[job])]
```


```python
positions
```




    [(34.1684364, -118.6058382),
     (40.7127281, -74.0060152),
     (35.2272086, -80.8430827),
     (40.7127281, -74.0060152),
     (41.5623178, -72.6509061)]




```python
calc_dist = lambda pos1, pos2:1/(geodesic(pos1, pos2).km+1)
```


```python
sim_matrix = create_similarity_matrix(calc_dist, positions)
```


```python
eval_similarity(sim_matrix, labels, k=0)
```


​    
![png](res/word2vec-job-posts-similarity_104_0.png)
​    


Now evaluate it


```python
pos_ = [["freiburg", "germany"], ["emmendingen", "germany"], ["offenburg", "germany"], ["berlin", "germany"]]
labels = ["freiburg", "emmendingen", "offenburg", "berlin"]
pos = []
for city, country in pos_:
    pos += [find_pos(city, country)]
```


```python
sim_matrix = create_similarity_matrix(calc_dist, pos)
```


```python
eval_similarity(sim_matrix, labels, k=0)
```


​    
![png](res/word2vec-job-posts-similarity_108_0.png)
​    


**Ergebnis:**

Mit diesem Verfahren lassen sich nun die Distanzen berechnen und so nahe Jobangebote finden, falls das gesucht ist. Dabei scheint das Verfahren so funktionieren, wie erhofft.<br>
<bR>
Die Evaluierung zeigt zudem, dass dieser Prozess auch wiklich funktioniert. Freiburg und Emmendingen sind sich am nähesten. Emmendingen ist näher an Offenburg als Freiburg. Und Berlin ist von allen drein weit weg. 

---

### **Experiment 9:** Ähnlichkeit der Job-Art

In diesem Experiment soll die Ähnlichkeit der Stellenausschreibungen mithilfe der Berufsart festgestellt werden.<br>Umgesetzt durch Regeln.

[<img src="./rackete_1.png" style="float:right" width=100></img>](#Table-of-Content)

Load the types.


```python
job1_type = data['job_type'][0]    # farmer
job2_type = data['job_type'][3]    # finance -> inevsting, black rock
job3_type = data['job_type'][4]    # data science -> analytics
job4_type = data['job_type'][2]    # data science -> machine learning
job5_type = data['job_type'][1]

types_ = [job1_type, job2_type, job3_type, job4_type, job5_type]
labels = ["job1", "job2", "job3", "job4", "job5"]
```


```python
data['job_type'].value_counts()
```




    Undefined     6109
    Full Time     3395
    Contract       488
    Part Time        6
    Internship       2
    Name: job_type, dtype: int64




```python
def type_sim(type1, type2):
    res = -1
    if type1 == "Undefined" or type2 == "Undefined":
        res = -1
    elif type1 == type2:
        res = 1.0
    elif (type1 == "Full Time" and type2 == "Contract") or (type2 == "Full Time" and type1 == "Contract"):
        res = -1
    elif (type1 == "Part Time" and type2 == "Full Time") or (type2 == "Part Time" and type1 == "Full Time"):
        res = 0.5
    elif (type1 == "Part Time" and type2 == "Internship") or (type2 == "Part Time" and type1 == "Internship"):
        res = 0.2
    elif (type1 == "Full Time" and type2 == "Internship") or (type2 == "Full Time" and type1 == "Internship"):
        res = 0.0
    elif (type1 == "Contract" and type2 == "Internship") or (type2 == "Contract" and type1 == "Internship"):
        res = -1
    return res
```

Evaluate


```python
sim_matrix = create_similarity_matrix(type_sim, types_)
```


```python
eval_similarity(sim_matrix, labels)
```


​    
![png](res/word2vec-job-posts-similarity_118_0.png)
​    


**Ergebnis:**

Durch den regelbasierten Ansatz kann man leicht die Berufe mit der Ähnlichkeit ihres Berufstyps bestimmen. Leider sind jedoch nicht viele gelabelt, was bei der Anwendung bedacht werden muss. 

---

### **Experiment 10:** Point-Gain-Proceedings

[<img src="./rackete_1.png" style="float:right" width=100></img>](#Table-of-Content)

Es sollen mehere Ansätze gewichtet kombiniert werden. Dies resultiert daraus, dass angenommen wird, dass die Ähnlichkeit subjektiv ist und somit variieren kann (Schwerpunkte festgelegt werden können).<br>
So ein Ansatz soll hier evaluiert werden.<br>
<br>
Bei dem Ansatz werden Punkte vergeben und umso mehr Punkte ein Job hat, desto ähnlicher soll dieser dem Zieljob sein. Dabei ist es nicht so schlimm, dass in manchen Kategorien kein Wert steht, da es hierfür nun einfach keine zusätzlichen Punkte gibt und das Jobausschreibung trotzdem noch als ähnlich gesehen werden kann.<br>
<br>
Außerdem kann man durch dieses Verfahren leicht Schwerpunkte setzen.<br>
<br>
Dieses Experiment wird direkt so programmiert, dass es später leicht einzusetzen ist.
<br>
<br>
> Vorgehen = Jede Kategorie gibt 0-5 Punkte und mithilfe der Parameter kann man diese Punktzahl gewichten um Schwerpunkte zu setzen

Eingabe ist eine Liste mit folgenden Einträgen und die Liste repräsentiert ein Job Post:

<img src="./cols.png" width=170></img>

Programmierung der Komponenten und des Algorithmus:


```python
def job_title_points(nlp, title1, title2):
    doc1 = nlp(title1)
    doc2 = nlp(title2)
    sim = doc1.similarity(doc2)
    
    if sim >= 0.95:
        return 5
    elif sim >= 0.9:
        return 4
    elif sim >= 0.8:
        return 2
    elif sim >= 0.7:
        return 1
    else:
        return 0
```


```python
def job_category_points(nlp, category1, category2, description1, description2):
    # fix the category if it nothing
    if type(category1) == float:
        category1 = get_most_common_noun(description1)
        
    if type(category2) == float:
        category2 = get_most_common_noun(description2)
        
    # build doc
    doc1 = nlp(category1)
    doc2 = nlp(category2)
    
    # calc similarity
    sim = doc1.similarity(doc2)
    
    if sim >= 0.95:
        return 5
    elif sim >= 0.9:
        return 3
    elif sim >= 0.8:
        return 1
    else:
        return 0
```


```python
def job_type_points(type1, type2):
    res = 0
    if type1 == "Undefined" or type2 == "Undefined":
        res = 0
    elif type1 == type2:
        res = 5
    elif (type1 == "Full Time" and type2 == "Contract") or (type2 == "Full Time" and type1 == "Contract"):
        res = 0
    elif (type1 == "Part Time" and type2 == "Full Time") or (type2 == "Part Time" and type1 == "Full Time"):
        res = 2
    elif (type1 == "Part Time" and type2 == "Internship") or (type2 == "Part Time" and type1 == "Internship"):
        res = 1
    elif (type1 == "Full Time" and type2 == "Internship") or (type2 == "Full Time" and type1 == "Internship"):
        res = 0
    elif (type1 == "Contract" and type2 == "Internship") or (type2 == "Contract" and type1 == "Internship"):
        res = 0
    return res
```


```python
def job_location_points(city1, country1, city2, country2):
    # Initialize Nominatim API
    geolocator = Nominatim(user_agent="MyApp")

    location1 = geolocator.geocode(f"{city1} {country1}", language="en")
    location2 = geolocator.geocode(f"{city2} {country2}", language="en")
    
    if location1 == None or location2 == None:
        return 0
    
    pos1 = (location1.latitude, location1.longitude)
    pos2 = (location2.latitude, location2.longitude)
    
    sim = 1 / (geodesic(pos1, pos2).km+1)
    
    if sim >= 0.1:
        return 5
    elif sim >= 0.07:
        return 4
    elif sim >= 0.03:
        return 3
    elif sim >= 0.01:
        return 1
    else:
        return 0
```


```python
# all categories gets between 0-5 points
def get_similar_job_posts(job_posts:pd.DataFrame, job_post:list, min_points=5, pruning=False, \
                                  title_w=2.0, category_w=1.0, type_w=1.0, pos_w=0.5, printing=True):
    
    # load other job posts 
    #all_job_posts = pd.read_excel("../data_scientist_united_states_job_postings_jobspikr.xlsx")
    all_job_posts = job_posts
    # create score-list
    #all_job_posts['score'] = 0.0
    score = np.array([0]*len(all_job_posts))
    
    # calc points
    nlp = spacy.load("en_core_web_lg")
    for post_idx in range(len(all_job_posts)):
        if printing: print(f"Calculate post {post_idx}...")
        # points for job-title similarity
        if title_w != 0:
            score[post_idx] += job_title_points(nlp, job_post[2], all_job_posts.loc[post_idx, :]['job_title']) * title_w
        
        # pruning -> if 0 points at the first, than skip
        if pruning and score[post_idx] == 0:
            continue
            
        # points for job-category similarity
        if category_w != 0:
            score[post_idx] += job_category_points(nlp, job_post[3], all_job_posts['category'][post_idx], \
                                               job_post[12], all_job_posts['job_description'][post_idx]) * category_w
        
        # points for job-type similarity  
        if type_w != 0:
            score[post_idx] += job_type_points(job_post[13], all_job_posts['job_type'][post_idx]) * type_w
        
        
        # points for job-location similarity  
        if pos_w != 0:
            score[post_idx] += job_location_points(job_post[5], job_post[7], all_job_posts['city'][post_idx], \
                                               all_job_posts['country'][post_idx]) * pos_w
        
    
    
    
    # return all posts with more than x points
    all_job_posts.loc[:, ['score']] = score
    return all_job_posts[all_job_posts['score'] >= min_points].sort_values(by="score", ascending=False)
```

**Evaluation**


```python
#post_id = np.random.randint(0, len(data))
post_id = 33
post = data.values.tolist()[post_id]
post
```




    [Timestamp('2019-02-06 08:30:12'),
     'https://www.careerbuilder.com/job/J3M2F963CC5895ZY3P0',
     'Senior Data Scientist - Tallahassee, FL - $150k-$170k',
     'business and financial operations',
     'Jefferson Frank',
     'Tallahassee',
     'FL',
     'Usa',
     'Tallahassee',
     'Florida',
     'Usa',
     Timestamp('2019-02-05 00:00:00'),
     'My client is a leader in the Manufacturing vertical and has operations in multiple states across the US. They are seeking to hire a full-time Senior Data Scientist to collaborate and work with their R&D, IT, Product Support, and Sales teams. Ideal Candidates Will Have: -Multiple years of Business Intelligence with the ability to work with structured and unstructured data -Multiple years of programming with C, C++, Java, or JavaScript languages -Experience working with AWS Services such as Redshift, S3, Athena, Kinesis My client has already begun interviewing candidates and is seeking to hire quickly. If interested call Mike @ 813-437-6882 and email your CV to m.greco@jeffersonfrank.com Jefferson Frank is the global leader for niche IT recruitment, advertising more Technology jobs than any other agency. We deal with both Partners & End Users throughout North America. By specializing solely in placing niche IT candidates in the market I have built relationships with most of the key employers in North America and have an unrivalled understanding of where the best opportunities and Microsoft jobs are. I understand the need for discretion and would welcome the opportunity to speak to any IT candidates that are considering a new career or job either now or in the future. Confidentiality is of course guaranteed. For information on the market and some of the opportunities that are available I can be contacted on 813-437-6882.',
     'Full Time',
     nan,
     'careerbuilder',
     'usa',
     1549440025100072,
     nan,
     '813 437 6882',
     'd33577ea9ae09c58d77e1fab2c012ba2',
     nan]




```python
#data.sample(n=50, replace=False)
#res = get_similar_job_posts(data.head(50), post, title_w=2.0, category_w=1.0, type_w=1.0, pos_w=0.5)
```


```python
#res.head()
```


```python
#get_similar_job_posts(data.head(50), post, title_w=2.0, category_w=0.0, type_w=1.0, pos_w=0.5, printing=False).head()
```

**Ergebnis:**

Das Punktesystem scheint zu funktionieren und es ist möglich seine Auswahl zu gewichten und damit andere optimale Jobangebote zu finden. <br>
Im obigen Beispiel wurden ebenfalls erst Data Science Stellenanzeigen gezeigt, welche im Bereich der Finanzen sind. Und bei Reduzierung der Gewichtung, war dies nicht mehr so. Somit hat der Ansatz funktioniert.<br>
<br>
An dieser Variante ist ebenfalls schön, dass sie leicht zu erweitern ist.<br>
<br>
Leider hat sich das Verfahren als sehr unperformant erwiesen. Im Experiment wurde ersichtlich, dass dies nicht nur an den Berechnungen an sich liegt, sondern vor allem an der großen Anzahl an Job-Posts.

---

### Experiment 11: Paralleles Ausführen
[<img src="./rackete_1.png" style="float:right" width=100></img>](#Table-of-Content)
Ziel ist es die Ausführung performanter zu gestalten. Hierzu 2 Gedanken:
- Reduzierung der Daten (random Wahl)
- Paralleles Ausführen

In diesem Experiment wird die Performancesteigerung durch Multiprocessing evaluiert.

1. Umschreiben des Algorithmus, für Prallelisierung:


```python
def calc_points(job_posts:pd.DataFrame, job_post, nlp, pruning, title_w, category_w, type_w, pos_w, printing):
    
    # create score-list
    score = np.array([0]*len(job_posts))
    
    for post_idx in range(len(job_posts)):
        #if printing: print(f"Calculate post {post_idx}...")
        # points for job-title similarity
        if title_w != 0:
            score[post_idx] += job_title_points(nlp, job_post[2], job_posts.loc[post_idx, :]['job_title']) * title_w

        # pruning -> if 0 points at the first, than skip
        if pruning and score[post_idx] == 0:
            continue

        # points for job-category similarity
        if category_w != 0:
            score[post_idx] += job_category_points(nlp, job_post[3], job_posts['category'][post_idx], \
                                               job_post[12], job_posts['job_description'][post_idx]) * category_w

        # points for job-type similarity  
        if type_w != 0:
            score[post_idx] += job_type_points(job_post[13], job_posts['job_type'][post_idx]) * type_w


        # points for job-location similarity  
        if pos_w != 0:
            score[post_idx] += job_location_points(job_post[5], job_post[7], job_posts['city'][post_idx], \
                                               job_posts['country'][post_idx]) * pos_w
    # return all posts with more than x points
    job_posts.loc[:, ['score']] = score
    log(f"One Process finished!", printing)
    return job_posts
```


```python
def log(txt:str, should_show=False):
    if should_show: print(txt)
```


```python
# all categories gets between 0-5 points
def get_similar_job_posts_parallel(job_posts:pd.DataFrame, job_post:list, min_points=5, pruning=False, \
                                  title_w=2.0, category_w=1.0, type_w=1.0, pos_w=0.5, printing=True):
    log_sym = "x"
    # load other job posts 
    all_job_posts = job_posts
    
    # open pool
    log(f"Starting {mp.cpu_count()} processes...", printing)
    pool = mp.Pool(mp.cpu_count())
    log(log_sym, printing)
    
    # split
    n = pool._processes
    log(f"Splitting data into {n} portions...", printing)
    max_ = len(all_job_posts)//n
    job_post_portions = []
    pointer = 0
    for i in range(n):
        job_post_portions += [all_job_posts.iloc[pointer:pointer+max_, :]]
        pointer += len(all_job_posts)//n
    log(log_sym, printing)
    log(f"Each portion contains {max_} jobposts...", printing)
     
    
    # calc points
    log(f"Loading SpaCy en_core_web_lg corpus...", printing)
    nlp = spacy.load("en_core_web_lg")
    log(log_sym, printing)
    
    # start processes / calc parallel the points / similarity
    log(f"Starts parallel calculation of the similarity/points...", printing)
    args = (job_post, nlp, pruning, title_w, category_w, type_w, pos_w, printing)
    results = [pool.apply(calc_points, args=(jobs,)+args) for jobs in job_post_portions]
    log(log_sym, printing)
    log(f"Finished with the parallel calculation of the similarity/points...", printing)

    # close mp pool
    pool.close() 
    
    # merge
    log(f"Merging scored job posts...", printing)
    scored_job_posts = results[0]
    for result in results[1:]:
        scored_job_posts.append(result, ignore_index=True)
    log(log_sym, printing)
    
    # take only important results and sort them
    log(f"Sorting scored job posts...", printing)
    r = scored_job_posts[scored_job_posts['score'] >= min_points].sort_values(by="score", ascending=False)
    log(log_sym, printing)
    return r
```


```python
#%%timeit -r 1 -n 1
#get_similar_job_posts_parallel(data.head(1000), post, title_w=2.0, category_w=0.0, \
#                                               type_w=1.0, pos_w=0.5, printing=True).head()
```

### Problem: Infinite runtime

Problem is that the multiprocessing didn't work on jupyter notebook but you can play around with writing in extra python file.

Trying with Dask.

https://examples.dask.org/applications/embarrassingly-parallel.html


```python
# all categories gets between 0-5 points
def get_similar_job_posts_parallel(job_posts, job_post:list, min_points=5, pruning=False, \
                                  title_w=2.0, category_w=1.0, type_w=1.0, pos_w=0.5, printing=True):
    log_sym = "x"
    # load other job posts 
    all_job_posts = job_posts
     
    
    # calc points
    log(f"Loading SpaCy en_core_web_lg corpus...", printing)
    nlp = spacy.load("en_core_web_lg")
    log(log_sym, printing)
    
    # start processes / calc parallel the points / similarity
    log(f"Starts parallel calculation of the similarity/points...", printing)
    args = (job_post, nlp, pruning, title_w, category_w, type_w, pos_w, printing)
    results = all_job_posts.apply(calc_points, axis=1, args=(jobs,)+args).compute()
    log(log_sym, printing)
    log(f"Finished with the parallel calculation of the similarity/points...", printing)
    
    # merge
    log(f"Merging scored job posts...", printing)
    scored_job_posts = results[0]
    for result in results[1:]:
        scored_job_posts.append(result, ignore_index=True)
    log(log_sym, printing)
    
    # take only important results and sort them
    log(f"Sorting scored job posts...", printing)
    r = scored_job_posts[scored_job_posts['score'] >= min_points].sort_values(by="score", ascending=False)
    log(log_sym, printing)
    return r
```


```python
excel_file = "../data_scientist_united_states_job_postings_jobspikr.xlsx"
parts = d.delayed(pd.read_excel)(excel_file)
df = dd.from_delayed(parts)

post_id = 33
post = data.values.tolist()[post_id]

posts = get_similar_job_posts_parallel(df, post, title_w=2.0, category_w=0.0, \
                                               type_w=1.0, pos_w=0.5, printing=True)
posts.head()
```

    Loading SpaCy en_core_web_lg corpus...
    x
    Starts parallel calculation of the similarity/points...



    ---------------------------------------------------------------------------
    
    TypeError                                 Traceback (most recent call last)
    
    File c:\Users\tobia\anaconda3\envs\ai\lib\site-packages\dask\dataframe\utils.py:177, in raise_on_meta_error(funcname, udf)
        <a href='file:///c%3A/Users/tobia/anaconda3/envs/ai/lib/site-packages/dask/dataframe/utils.py?line=175'>176</a> try:
    --> <a href='file:///c%3A/Users/tobia/anaconda3/envs/ai/lib/site-packages/dask/dataframe/utils.py?line=176'>177</a>     yield
        <a href='file:///c%3A/Users/tobia/anaconda3/envs/ai/lib/site-packages/dask/dataframe/utils.py?line=177'>178</a> except Exception as e:


    File c:\Users\tobia\anaconda3\envs\ai\lib\site-packages\dask\dataframe\core.py:6086, in _emulate(func, udf, *args, **kwargs)
       <a href='file:///c%3A/Users/tobia/anaconda3/envs/ai/lib/site-packages/dask/dataframe/core.py?line=6084'>6085</a> with raise_on_meta_error(funcname(func), udf=udf):
    -> <a href='file:///c%3A/Users/tobia/anaconda3/envs/ai/lib/site-packages/dask/dataframe/core.py?line=6085'>6086</a>     return func(*_extract_meta(args, True), **_extract_meta(kwargs, True))


    File c:\Users\tobia\anaconda3\envs\ai\lib\site-packages\dask\utils.py:1018, in methodcaller.__call__(self, _methodcaller__obj, *args, **kwargs)
       <a href='file:///c%3A/Users/tobia/anaconda3/envs/ai/lib/site-packages/dask/utils.py?line=1016'>1017</a> def __call__(self, __obj, *args, **kwargs):
    -> <a href='file:///c%3A/Users/tobia/anaconda3/envs/ai/lib/site-packages/dask/utils.py?line=1017'>1018</a>     return getattr(__obj, self.method)(*args, **kwargs)


    File c:\Users\tobia\anaconda3\envs\ai\lib\site-packages\pandas\core\frame.py:8833, in DataFrame.apply(self, func, axis, raw, result_type, args, **kwargs)
       <a href='file:///c%3A/Users/tobia/anaconda3/envs/ai/lib/site-packages/pandas/core/frame.py?line=8823'>8824</a> op = frame_apply(
       <a href='file:///c%3A/Users/tobia/anaconda3/envs/ai/lib/site-packages/pandas/core/frame.py?line=8824'>8825</a>     self,
       <a href='file:///c%3A/Users/tobia/anaconda3/envs/ai/lib/site-packages/pandas/core/frame.py?line=8825'>8826</a>     func=func,
       (...)
       <a href='file:///c%3A/Users/tobia/anaconda3/envs/ai/lib/site-packages/pandas/core/frame.py?line=8830'>8831</a>     kwargs=kwargs,
       <a href='file:///c%3A/Users/tobia/anaconda3/envs/ai/lib/site-packages/pandas/core/frame.py?line=8831'>8832</a> )
    -> <a href='file:///c%3A/Users/tobia/anaconda3/envs/ai/lib/site-packages/pandas/core/frame.py?line=8832'>8833</a> return op.apply().__finalize__(self, method="apply")


    File c:\Users\tobia\anaconda3\envs\ai\lib\site-packages\pandas\core\apply.py:727, in FrameApply.apply(self)
        <a href='file:///c%3A/Users/tobia/anaconda3/envs/ai/lib/site-packages/pandas/core/apply.py?line=724'>725</a>     return self.apply_raw()
    --> <a href='file:///c%3A/Users/tobia/anaconda3/envs/ai/lib/site-packages/pandas/core/apply.py?line=726'>727</a> return self.apply_standard()


    File c:\Users\tobia\anaconda3\envs\ai\lib\site-packages\pandas\core\apply.py:851, in FrameApply.apply_standard(self)
        <a href='file:///c%3A/Users/tobia/anaconda3/envs/ai/lib/site-packages/pandas/core/apply.py?line=849'>850</a> def apply_standard(self):
    --> <a href='file:///c%3A/Users/tobia/anaconda3/envs/ai/lib/site-packages/pandas/core/apply.py?line=850'>851</a>     results, res_index = self.apply_series_generator()
        <a href='file:///c%3A/Users/tobia/anaconda3/envs/ai/lib/site-packages/pandas/core/apply.py?line=852'>853</a>     # wrap results


    File c:\Users\tobia\anaconda3\envs\ai\lib\site-packages\pandas\core\apply.py:867, in FrameApply.apply_series_generator(self)
        <a href='file:///c%3A/Users/tobia/anaconda3/envs/ai/lib/site-packages/pandas/core/apply.py?line=864'>865</a> for i, v in enumerate(series_gen):
        <a href='file:///c%3A/Users/tobia/anaconda3/envs/ai/lib/site-packages/pandas/core/apply.py?line=865'>866</a>     # ignore SettingWithCopy here in case the user mutates
    --> <a href='file:///c%3A/Users/tobia/anaconda3/envs/ai/lib/site-packages/pandas/core/apply.py?line=866'>867</a>     results[i] = self.f(v)
        <a href='file:///c%3A/Users/tobia/anaconda3/envs/ai/lib/site-packages/pandas/core/apply.py?line=867'>868</a>     if isinstance(results[i], ABCSeries):
        <a href='file:///c%3A/Users/tobia/anaconda3/envs/ai/lib/site-packages/pandas/core/apply.py?line=868'>869</a>         # If we have a view on v, we need to make a copy because
        <a href='file:///c%3A/Users/tobia/anaconda3/envs/ai/lib/site-packages/pandas/core/apply.py?line=869'>870</a>         #  series_generator will swap out the underlying data


    File c:\Users\tobia\anaconda3\envs\ai\lib\site-packages\pandas\core\apply.py:138, in Apply.__init__.<locals>.f(x)
        <a href='file:///c%3A/Users/tobia/anaconda3/envs/ai/lib/site-packages/pandas/core/apply.py?line=136'>137</a> def f(x):
    --> <a href='file:///c%3A/Users/tobia/anaconda3/envs/ai/lib/site-packages/pandas/core/apply.py?line=137'>138</a>     return func(x, *args, **kwargs)


    TypeError: calc_points() takes 9 positional arguments but 10 were given


​    
    The above exception was the direct cause of the following exception:


    ValueError                                Traceback (most recent call last)
    
    d:\Studium\4. Semester\Module\NLP\Praktikum\2022_05_16 Job Posts Similarity\Word2Vec-Ansatz\word2vec-job-posts-similarity.ipynb Cell 142' in <cell line: 8>()
          <a href='vscode-notebook-cell:/d%3A/Studium/4.%20Semester/Module/NLP/Praktikum/2022_05_16%20Job%20Posts%20Similarity/Word2Vec-Ansatz/word2vec-job-posts-similarity.ipynb#ch0000151?line=4'>5</a> post_id = 33
          <a href='vscode-notebook-cell:/d%3A/Studium/4.%20Semester/Module/NLP/Praktikum/2022_05_16%20Job%20Posts%20Similarity/Word2Vec-Ansatz/word2vec-job-posts-similarity.ipynb#ch0000151?line=5'>6</a> post = data.values.tolist()[post_id]
    ----> <a href='vscode-notebook-cell:/d%3A/Studium/4.%20Semester/Module/NLP/Praktikum/2022_05_16%20Job%20Posts%20Similarity/Word2Vec-Ansatz/word2vec-job-posts-similarity.ipynb#ch0000151?line=7'>8</a> posts = get_similar_job_posts_parallel(df, post, title_w=2.0, category_w=0.0, \
          <a href='vscode-notebook-cell:/d%3A/Studium/4.%20Semester/Module/NLP/Praktikum/2022_05_16%20Job%20Posts%20Similarity/Word2Vec-Ansatz/word2vec-job-posts-similarity.ipynb#ch0000151?line=8'>9</a>                                                type_w=1.0, pos_w=0.5, printing=True)
         <a href='vscode-notebook-cell:/d%3A/Studium/4.%20Semester/Module/NLP/Praktikum/2022_05_16%20Job%20Posts%20Similarity/Word2Vec-Ansatz/word2vec-job-posts-similarity.ipynb#ch0000151?line=9'>10</a> posts.head()


    d:\Studium\4. Semester\Module\NLP\Praktikum\2022_05_16 Job Posts Similarity\Word2Vec-Ansatz\word2vec-job-posts-similarity.ipynb Cell 141' in get_similar_job_posts_parallel(job_posts, job_post, min_points, pruning, title_w, category_w, type_w, pos_w, printing)
         <a href='vscode-notebook-cell:/d%3A/Studium/4.%20Semester/Module/NLP/Praktikum/2022_05_16%20Job%20Posts%20Similarity/Word2Vec-Ansatz/word2vec-job-posts-similarity.ipynb#ch0000140?line=14'>15</a> log(f"Starts parallel calculation of the similarity/points...", printing)
         <a href='vscode-notebook-cell:/d%3A/Studium/4.%20Semester/Module/NLP/Praktikum/2022_05_16%20Job%20Posts%20Similarity/Word2Vec-Ansatz/word2vec-job-posts-similarity.ipynb#ch0000140?line=15'>16</a> args = (job_post, nlp, pruning, title_w, category_w, type_w, pos_w, printing)
    ---> <a href='vscode-notebook-cell:/d%3A/Studium/4.%20Semester/Module/NLP/Praktikum/2022_05_16%20Job%20Posts%20Similarity/Word2Vec-Ansatz/word2vec-job-posts-similarity.ipynb#ch0000140?line=16'>17</a> results = all_job_posts.apply(calc_points, axis=1, args=(jobs,)+args).compute()
         <a href='vscode-notebook-cell:/d%3A/Studium/4.%20Semester/Module/NLP/Praktikum/2022_05_16%20Job%20Posts%20Similarity/Word2Vec-Ansatz/word2vec-job-posts-similarity.ipynb#ch0000140?line=17'>18</a> log(log_sym, printing)
         <a href='vscode-notebook-cell:/d%3A/Studium/4.%20Semester/Module/NLP/Praktikum/2022_05_16%20Job%20Posts%20Similarity/Word2Vec-Ansatz/word2vec-job-posts-similarity.ipynb#ch0000140?line=18'>19</a> log(f"Finished with the parallel calculation of the similarity/points...", printing)


    File c:\Users\tobia\anaconda3\envs\ai\lib\site-packages\dask\dataframe\core.py:5236, in DataFrame.apply(self, func, axis, broadcast, raw, reduce, args, meta, result_type, **kwds)
       <a href='file:///c%3A/Users/tobia/anaconda3/envs/ai/lib/site-packages/dask/dataframe/core.py?line=5232'>5233</a>     raise NotImplementedError(msg)
       <a href='file:///c%3A/Users/tobia/anaconda3/envs/ai/lib/site-packages/dask/dataframe/core.py?line=5234'>5235</a> if meta is no_default:
    -> <a href='file:///c%3A/Users/tobia/anaconda3/envs/ai/lib/site-packages/dask/dataframe/core.py?line=5235'>5236</a>     meta = _emulate(
       <a href='file:///c%3A/Users/tobia/anaconda3/envs/ai/lib/site-packages/dask/dataframe/core.py?line=5236'>5237</a>         M.apply, self._meta_nonempty, func, args=args, udf=True, **kwds
       <a href='file:///c%3A/Users/tobia/anaconda3/envs/ai/lib/site-packages/dask/dataframe/core.py?line=5237'>5238</a>     )
       <a href='file:///c%3A/Users/tobia/anaconda3/envs/ai/lib/site-packages/dask/dataframe/core.py?line=5238'>5239</a>     warnings.warn(meta_warning(meta))
       <a href='file:///c%3A/Users/tobia/anaconda3/envs/ai/lib/site-packages/dask/dataframe/core.py?line=5239'>5240</a> kwds.update({"parent_meta": self._meta})


    File c:\Users\tobia\anaconda3\envs\ai\lib\site-packages\dask\dataframe\core.py:6086, in _emulate(func, udf, *args, **kwargs)
       <a href='file:///c%3A/Users/tobia/anaconda3/envs/ai/lib/site-packages/dask/dataframe/core.py?line=6080'>6081</a> """
       <a href='file:///c%3A/Users/tobia/anaconda3/envs/ai/lib/site-packages/dask/dataframe/core.py?line=6081'>6082</a> Apply a function using args / kwargs. If arguments contain dd.DataFrame /
       <a href='file:///c%3A/Users/tobia/anaconda3/envs/ai/lib/site-packages/dask/dataframe/core.py?line=6082'>6083</a> dd.Series, using internal cache (``_meta``) for calculation
       <a href='file:///c%3A/Users/tobia/anaconda3/envs/ai/lib/site-packages/dask/dataframe/core.py?line=6083'>6084</a> """
       <a href='file:///c%3A/Users/tobia/anaconda3/envs/ai/lib/site-packages/dask/dataframe/core.py?line=6084'>6085</a> with raise_on_meta_error(funcname(func), udf=udf):
    -> <a href='file:///c%3A/Users/tobia/anaconda3/envs/ai/lib/site-packages/dask/dataframe/core.py?line=6085'>6086</a>     return func(*_extract_meta(args, True), **_extract_meta(kwargs, True))


    File c:\Users\tobia\anaconda3\envs\ai\lib\contextlib.py:137, in _GeneratorContextManager.__exit__(self, typ, value, traceback)
        <a href='file:///c%3A/Users/tobia/anaconda3/envs/ai/lib/contextlib.py?line=134'>135</a>     value = typ()
        <a href='file:///c%3A/Users/tobia/anaconda3/envs/ai/lib/contextlib.py?line=135'>136</a> try:
    --> <a href='file:///c%3A/Users/tobia/anaconda3/envs/ai/lib/contextlib.py?line=136'>137</a>     self.gen.throw(typ, value, traceback)
        <a href='file:///c%3A/Users/tobia/anaconda3/envs/ai/lib/contextlib.py?line=137'>138</a> except StopIteration as exc:
        <a href='file:///c%3A/Users/tobia/anaconda3/envs/ai/lib/contextlib.py?line=138'>139</a>     # Suppress StopIteration *unless* it's the same exception that
        <a href='file:///c%3A/Users/tobia/anaconda3/envs/ai/lib/contextlib.py?line=139'>140</a>     # was passed to throw().  This prevents a StopIteration
        <a href='file:///c%3A/Users/tobia/anaconda3/envs/ai/lib/contextlib.py?line=140'>141</a>     # raised inside the "with" statement from being suppressed.
        <a href='file:///c%3A/Users/tobia/anaconda3/envs/ai/lib/contextlib.py?line=141'>142</a>     return exc is not value


    File c:\Users\tobia\anaconda3\envs\ai\lib\site-packages\dask\dataframe\utils.py:198, in raise_on_meta_error(funcname, udf)
        <a href='file:///c%3A/Users/tobia/anaconda3/envs/ai/lib/site-packages/dask/dataframe/utils.py?line=188'>189</a> msg += (
        <a href='file:///c%3A/Users/tobia/anaconda3/envs/ai/lib/site-packages/dask/dataframe/utils.py?line=189'>190</a>     "Original error is below:\n"
        <a href='file:///c%3A/Users/tobia/anaconda3/envs/ai/lib/site-packages/dask/dataframe/utils.py?line=190'>191</a>     "------------------------\n"
       (...)
        <a href='file:///c%3A/Users/tobia/anaconda3/envs/ai/lib/site-packages/dask/dataframe/utils.py?line=194'>195</a>     "{2}"
        <a href='file:///c%3A/Users/tobia/anaconda3/envs/ai/lib/site-packages/dask/dataframe/utils.py?line=195'>196</a> )
        <a href='file:///c%3A/Users/tobia/anaconda3/envs/ai/lib/site-packages/dask/dataframe/utils.py?line=196'>197</a> msg = msg.format(f" in `{funcname}`" if funcname else "", repr(e), tb)
    --> <a href='file:///c%3A/Users/tobia/anaconda3/envs/ai/lib/site-packages/dask/dataframe/utils.py?line=197'>198</a> raise ValueError(msg) from e


    ValueError: Metadata inference failed in `apply`.
    
    You have supplied a custom function and Dask is unable to 
    determine the type of output that that function returns. 
    
    To resolve this please provide a meta= keyword.
    The docstring of the Dask function you ran should have more information.
    
    Original error is below:
    ------------------------
    TypeError('calc_points() takes 9 positional arguments but 10 were given')
    
    Traceback:
    ---------
      File "c:\Users\tobia\anaconda3\envs\ai\lib\site-packages\dask\dataframe\utils.py", line 177, in raise_on_meta_error
        yield
      File "c:\Users\tobia\anaconda3\envs\ai\lib\site-packages\dask\dataframe\core.py", line 6086, in _emulate
        return func(*_extract_meta(args, True), **_extract_meta(kwargs, True))
      File "c:\Users\tobia\anaconda3\envs\ai\lib\site-packages\dask\utils.py", line 1018, in __call__
        return getattr(__obj, self.method)(*args, **kwargs)
      File "c:\Users\tobia\anaconda3\envs\ai\lib\site-packages\pandas\core\frame.py", line 8833, in apply
        return op.apply().__finalize__(self, method="apply")
      File "c:\Users\tobia\anaconda3\envs\ai\lib\site-packages\pandas\core\apply.py", line 727, in apply
        return self.apply_standard()
      File "c:\Users\tobia\anaconda3\envs\ai\lib\site-packages\pandas\core\apply.py", line 851, in apply_standard
        results, res_index = self.apply_series_generator()
      File "c:\Users\tobia\anaconda3\envs\ai\lib\site-packages\pandas\core\apply.py", line 867, in apply_series_generator
        results[i] = self.f(v)
      File "c:\Users\tobia\anaconda3\envs\ai\lib\site-packages\pandas\core\apply.py", line 138, in f
        return func(x, *args, **kwargs)



**Evaluation**


```python

```

---
(sementic)
description similarity [see here](https://towardsdatascience.com/semantic-similarity-using-transformers-8f3cb5bf66d6) or [here](https://www.sbert.net/docs/usage/semantic_textual_similarity.html)

---

### Evaluation

[<img src="./rackete_1.png" style="float:right" width=100></img>](#Table-of-Content)

Die Berechnung der Ähnlichlichkeit über die Ähnlichkeit der Jobbeschreibung hat nicht so funktioniert wie gewünscht. Selbst mit Anpassungen scheint es nicht gut die Ähnlichkeit berechnen zu können.<br>
Dafür hat dies über den Job-Titel und der Word2Vec-Technik sehr gut funktioniert.<br>
<br>
Wie das Experiment 10 gezeigt hat, kann man dieses Verfahren in Verbindung mit anderen Ähnlichkeiten einsetzen, um so ein individuelles Ähnlickeitsmaß herstellen zu können. <br>
<br>
Ein größeres Problem ist die Performance. Es existieren nämlich sehr viele Job-Posts und um alle zu verwenden, wird viel Zeit in Anspruch genommen.<br>
Als Lösungsansatz könnte man zum einen nicht alle Daten verwenden (zufällig 1/4 der Daten) und zum anderen könnte der Prozess der Punkteberechnung in 2 oder mehr parallele Prozesse geteilt werden.

---
### Anwendung

[<img src="./rackete_1.png" style="float:right" width=100></img>](#Table-of-Content)


```python
def main():
    #data = pd.read_excel("./data_scientist_united_states_job_postings_jobspikr.xlsx")
    data = ds.read_excel("./data_scientist_united_states_job_postings_jobspikr.xlsx")
    choose_a_post = False
    while not choose_a_post:
        post_id = get_number_input("Choose one number to choose a job post ", 0, data.shape[0]-1)
        
        post = data.values.tolist()[post_id]
        print_job_post(post)
        answer = get_number_input("Is this ok? (1=yes / 0=no)", 0, 1)
        if answer == 1:
            choose_a_post = True


    posts = get_similar_job_posts_parallel(data.head(1000), post, title_w=2.0, category_w=0.0, \
                                            type_w=1.0, pos_w=0.5, printing=True).head()


    cur_idx = 0
    print("-----\nNavigate with 'next', 'prev', 'exit'\n-----")
    while True:
        print_job_post(posts[cur_idx])
        user_input = input("User:")
        if user_input == "next":
            if cur_idx < posts.shape[0]-1:
                cur_idx += 1
        elif user_input == "prev":
            if cur_idx > 0:
                cur_idx -= 1
        elif user_input == "exit":
            print("bye")
            break
```

---
