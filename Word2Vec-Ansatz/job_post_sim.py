import sys
import time

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
from threading import Thread



def get_most_common_noun(job_description, nlp):
    #nlp = spacy.load("en_core_web_lg")
    doc = nlp(job_description)
    words = dict()
    for token in doc:
        if token.pos_ in ["NOUN", "PROPN"]:
            if token.text in words.keys():
                words[token.text] += 1
            else:
                words[token.text] = 1
    return sorted(words.items(), key=lambda x:x[0])[0][0]           



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



def job_category_points(nlp, category1, category2, description1, description2):
    # fix the category if it nothing
    if type(category1) == float:
        category1 = get_most_common_noun(description1, nlp)
        
    if type(category2) == float:
        category2 = get_most_common_noun(description2, nlp)
        
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



def job_type_points(type1, type2):
    res = 0.0    

    if type1 == type2:
        res = 1.0
    elif (type1 == "Full Time" and type2 == "Contract") or (type2 == "Full Time" and type1 == "Contract"):
        res = 0.5
    elif (type1 == "Part Time" and type2 == "Full Time") or (type2 == "Part Time" and type1 == "Full Time"):
        res = 0.5
    elif (type1 == "Part Time" and type2 == "Internship") or (type2 == "Part Time" and type1 == "Internship"):
        res = 0.2
    elif (type1 == "Part Time" and type2 == "Contract") or (type2 == "Part Time" and type1 == "Contract"):
        res = 0.5
    elif (type1 == "Full Time" and type2 == "Internship") or (type2 == "Full Time" and type1 == "Internship"):
        res = 0.0
    elif (type1 == "Contract" and type2 == "Internship") or (type2 == "Contract" and type1 == "Internship"):
        res = 0.5
    elif type1 == "Undefined" or type2 == "Undefined":
        res = 0.1
    return res



def job_location_points(city1, country1, city2, country2, weight):
    # Initialize Nominatim API
    geolocator = Nominatim(user_agent="MyApp")

    location1 = geolocator.geocode(f"{city1} {country1}", language="en", timeout=10000)
    location2 = geolocator.geocode(f"{city2} {country2}", language="en", timeout=10000)
    
    if location1 == None or location2 == None:
        return 0
    
    pos1 = (location1.latitude, location1.longitude)
    pos2 = (location2.latitude, location2.longitude)
    
    sim = 1 / (geodesic(pos1, pos2).km+1)

    if weight < 0:
        sim = revert_sim(sim)
    
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


def log(txt:str, should_show=False):
    if should_show: print("\n"+txt+"\n")


def calc_points(job_posts:pd.DataFrame, job_post, progress, total, nlp, pruning, title_w, category_w, type_w, pos_w, printing):
    
    # create score-list
    score = np.array([0]*len(job_posts))
    
    for post_idx in range(len(job_posts)):  #
        #if printing: print(f"Calculate post {post_idx}...")
        # points for job-title similarity
        # old: job_posts['job_description'][post_idx]
        if title_w != 0:
            score[post_idx] += job_title_points(nlp, job_post[2], job_posts.iloc[post_idx, 2]) * title_w

        # pruning -> if 0 points at the first, than skip
        if pruning and score[post_idx] == 0:
            continue

        # points for job-category similarity
        if category_w != 0:
            score[post_idx] += job_category_points(nlp, job_post[3], job_posts.iloc[post_idx, 3], \
                                            job_post[12], job_posts.iloc[post_idx, 12]) * category_w

        # points for job-type similarity  
        if type_w != 0:
            score[post_idx] += job_type_points(job_post[13], job_posts.iloc[post_idx, 13]) * type_w


        # points for job-location similarity  
        if pos_w != 0:
            print("WARNING: CALC LOCATION IS EXPENSIVE!")
            score[post_idx] += job_location_points(job_post[5], job_post[7], job_posts.iloc[post_idx, 5], \
                                           job_posts.iloc[post_idx, 7], weight) * abs(pos_w)

        progress[0] += 1
        progress_bar(progress[0], total)

    # return all posts with more than x points
    job_posts.loc[:, ['score']] = score
    log(f"One Process finished!", printing)
    return job_posts


# all categories gets between 0-5 points
def get_similar_job_posts(job_posts:pd.DataFrame, job_post:list, min_points=5, pruning=False, \
                                  title_w=2.0, category_w=1.0, type_w=1.0, pos_w=0.5, printing=True):
    log_sym = "x"
    # load other job posts 
    all_job_posts = job_posts
    
    # calc points
    log(f"Loading SpaCy en_core_web_lg corpus...", printing)
    nlp = spacy.load("en_core_web_lg")
    log(log_sym, printing)
    
    # start processes / calc parallel the points / similarity
    log(f"Starts calculation of the similarity/points...", printing)
    progress = [0]    # use this for changing
    total = job_posts.shape[0]
    args = (job_post, progress, total, nlp, pruning, title_w, category_w, type_w, pos_w, printing)
    scored_job_posts = calc_points(all_job_posts, *args)
    log(log_sym, printing)
    
    # sort and return
    if len(scored_job_posts) > 0:
        # take only important results and sort them
        log(f"Sorting scored job posts...", printing)
        r = scored_job_posts[scored_job_posts['score'] >= min_points].sort_values(by="score", ascending=False)
        log(log_sym, printing)
        return r



def calc_points_parallel(result, job_posts:pd.DataFrame, job_post, progress, total, nlp, pruning, title_w, category_w, type_w, pos_w, printing):
    
    # create score-list
    score = np.array([0]*len(job_posts))
    
    for post_idx in range(len(job_posts)):
        #if printing: print(f"Calculate post {post_idx}...")
        # points for job-title similarity
        if title_w != 0:
            score[post_idx] += job_title_points(nlp, job_post[2], job_posts.loc[post_idx+offset, :]['job_title']) * title_w

        # pruning -> if 0 points at the first, than skip
        if pruning and score[post_idx] == 0:
            continue

        # points for job-category similarity
        if category_w != 0:
            score[post_idx] += job_category_points(nlp, job_post[3], job_posts['category'][post_idx+offset], \
                                            job_post[12], job_posts['job_description'][post_idx+offset]) * category_w

        # points for job-type similarity  
        if type_w != 0:
            score[post_idx] += job_type_points(job_post[13], job_posts['job_type'][post_idx+offset]) * type_w


        # points for job-location similarity  
        if pos_w != 0:
            score[post_idx] += job_location_points(job_post[5], job_post[7], job_posts['city'][post_idx+offset], \
                                            job_posts['country'][post_idx+offset], weight) * abs(pos_w)

        progress[0] += 1
        progress_bar(progress[0], total)

    # return all posts with more than x points
    job_posts.loc[:, ['score']] = score
    log(f"One Process finished!", printing)
    result += [job_posts]
    #return job_posts


# all categories gets between 0-5 points
def get_similar_job_posts_parallel(job_posts:pd.DataFrame, job_post:list, min_points=5, pruning=False, \
                                  title_w=2.0, category_w=1.0, type_w=1.0, pos_w=0.5, printing=True, thread_amount=1):
    log_sym = "x"
    # load other job posts 
    all_job_posts = job_posts
    
    # split
    n = thread_amount   #mp.cpu_count()
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
    log(f"Starts parallel calculation of the similarity/points with {n} Threads...", printing)
    progress = [0]    # use this for changing
    total = job_posts.shape[0]
    args = (job_post, progress, total, nlp, pruning, title_w, category_w, type_w, pos_w, printing)
    results = []
    threads = []
    for jobs in job_post_portions:
        t = Thread(target=calc_points_parallel, args=(results, jobs,)+args)
        threads += [t]
        t.start()
    #log(log_sym, printing)
    #log(f"Created Threads and they running...\n", printing)

    # wait until finishes
    #log(f"Waiting for finishing tasks...", printing)
    for t in threads: 
        t.join()
    log(log_sym, printing)
    log(f"Finished with the parallel calculation of the similarity/points...\n", printing)
    
    # merge
    log(f"Merging scored job posts...", printing)
    if len(results) > 0:
        scored_job_posts = results[0]
        for result in results[1:]:
            scored_job_posts.append(result, ignore_index=True)
        log(log_sym, printing)
        
        # take only important results and sort them
        log(f"Sorting scored job posts...", printing)
        r = scored_job_posts[scored_job_posts['score'] >= min_points].sort_values(by="score", ascending=False)
        log(log_sym, printing)
        return r


def get_number_input(msg:str, min, max):
    wrong_input = True
    while wrong_input:
        #addition = ""
        #if min != None:
        #    addition += f"(min={min})"
        #if max != None:
        #    if min != None:
        #        addition += ", "
        #    addition += f"(max={max})"
        user_input = input(f"{msg}:")#+addition

        if user_input == "exit":
            sys.exit()

        try:
            result = int(user_input)
            if result >= min and result <= max:
                wrong_input = False
            else:
                print("Try again. Type a number.")# + addition)
        except ValueError:
            pass
    return result

def get_float_number_input(msg:str, min:float, max:float):
    wrong_input = True
    while wrong_input:
        user_input = input(f"{msg}:")

        if user_input == "exit":
            sys.exit()
        elif user_input == "":
            return None

        try:
            result = float(user_input)
            if result >= min and result <= max:
                wrong_input = False
            else:
                print("Try again. Type a float number.")
        except ValueError:
            pass
    return result


def get_bool_input(msg:str):
    wrong_input = True
    while wrong_input:
        user_input = input(f"{msg}:")

        if user_input == "exit":
            sys.exit()
        elif user_input in ["y", "", "yes", "1"]:
            result = 1
            wrong_input = False
        elif user_input in ["no", "0", "n"]:
            result = 0
            wrong_input = False

    return result


def print_job_post(job_post, with_r=False):
    width = 64
    job_str = ""
    if with_r:
        job_str += "\r"
    job_str += f"\n{'-'*width}\n{' '*((width//2)-8)}>>> Job Post <<<\n{'-'*width}\n"
    job_str += f"\n\nTitle: {job_post[2]}\n"
    job_str += f"\nCategory: {job_post[3]}\n"
    job_str += f"\nLocation: {job_post[5]} in {job_post[7]}\n"
    job_str += f"\nType: {job_post[13]}\n"
    job_str += f"\nDespription:\n\n {'-'*width}\n\n{job_post[12]}\n{'-'*width}\n"
    if with_r:
        print(job_str, end="\r")
    else:
        print(job_str)


def progress_bar(progress, total):
    percentage = 100 * (progress/float(total))
    bar = '#'*int(percentage) + '-'*(100-int(percentage))
    print(f"\r[{bar}] {percentage:.2f}%", end="\r")


def calc_points_experiment(result, job_posts:pd.DataFrame, job_post, time_, progress, total, nlp, pruning, title_w, category_w, type_w, pos_w, printing):
    
    # create score-list
    score = np.array([0]*len(job_posts))

    offset = job_posts.index.start
    
    for post_idx in range(len(job_posts)):
        #if printing: print(f"Calculate post {post_idx}...")
        # points for job-title similarity
        start = time.time()
        if title_w != 0:
            score[post_idx] += job_title_points(nlp, job_post[2], job_posts.loc[post_idx+offset, :]['job_title']) * title_w
        duration = round(time.time() - start, 2)
        time_['title'] += duration

        # pruning -> if 0 points at the first, than skip
        if pruning and score[post_idx] == 0:
            continue

        # points for job-category similarity
        start = time.time()
        if category_w != 0:
            score[post_idx] += job_category_points(nlp, job_post[3], job_posts['category'][post_idx+offset], \
                                            job_post[12], job_posts['job_description'][post_idx+offset]) * category_w
        duration = round(time.time() - start, 2)
        time_['category'] += duration

        # points for job-type similarity  
        start = time.time()
        if type_w != 0:
            score[post_idx] += job_type_points(job_post[13], job_posts['job_type'][post_idx+offset]) * type_w
        duration = round(time.time() - start, 2)
        time_['type'] += duration

        # points for job-location similarity  
        start = time.time()
        if pos_w != 0:
            score[post_idx] += job_location_points(job_post[5], job_post[7], job_posts['city'][post_idx+offset], \
                                            job_posts['country'][post_idx+offset]) * pos_w
        duration = round(time.time() - start, 2)
        time_['location'] += duration


        progress[0] += 1
        progress_bar(progress[0], total)

    # return all posts with more than x points
    job_posts.loc[:, ['score']] = score
    log(f"One Process finished!", printing)
    result += [job_posts]



# all categories gets between 0-5 points
def get_similar_job_posts_experiment(job_posts:pd.DataFrame, job_post:list, min_points=5, pruning=False, \
                                  title_w=2.0, category_w=1.0, type_w=1.0, pos_w=0.5, printing=True, thread_amount=1):
    time_ = {'title':0, 'location':0, 'type':0, 'category':0}
    # load other job posts 
    all_job_posts = job_posts
    
    # split
    n = thread_amount   
    max_ = len(all_job_posts)//n
    job_post_portions = []
    pointer = 0
    for i in range(n):
        job_post_portions += [all_job_posts.iloc[pointer:pointer+max_, :]]
        pointer += len(all_job_posts)//n
     
    # calc points
    nlp = spacy.load("en_core_web_lg")

    # start processes / calc parallel the points / similarity
    progress = [0]    # use this for changing
    total = job_posts.shape[0]
    args = (job_post, time_, progress, total, nlp, pruning, title_w, category_w, type_w, pos_w, printing)
    results = []
    threads = []
    for jobs in job_post_portions:
        t = Thread(target=calc_points_experiment, args=(results, jobs,)+args)
        threads += [t]
        t.start()

    # wait until finishes
    for t in threads: 
        t.join()
    
    # merge
    if len(results) > 0:
        scored_job_posts = results[0]
        for result in results[1:]:
            scored_job_posts.append(result, ignore_index=True)
        
        # take only important results and sort them
        r = scored_job_posts[scored_job_posts['score'] >= min_points].sort_values(by="score", ascending=False)
    return f"   **title:** {time_['title']} seconds<br>\n   **location:** {time_['location']} seconds<br>\n   **type:** {time_['type']} seconds<br>\n   **category:** {time_['category']} seconds<br>\n\n"


def time_experiment():
    data = pd.read_excel("./data_scientist_united_states_job_postings_jobspikr.xlsx")
    post = data.values.tolist()[33]

    results = ""

    with open("./time_experiment_reults.txt", "w") as file:
            file.write(" ")

    for i in [10, 100, 1000]:  # 10000
        print(f"Startet {i} samples...")
        results += f"{'-'*64}\nSample Size = {i}\n\n{'-'*64}\n"
        for n in [1, 2, 5, 10]:
            print("Waiting for geopy...")
            time.sleep(10)
            print(f"Try on {n} Threads...")
            start = time.time()
            output = get_similar_job_posts_experiment(data.head(i), post, title_w=2.0, category_w=0.0, type_w=1.0, pos_w=0.5, printing=False, thread_amount=n)
            duration = time.time()-start
            results += f"\n**Threads:** {n}\n**Time:**{duration:.2f} seconds<br>\n{output}<br><br>\n"
            print(f"\nThreads: {n} needed {duration:.2f} seconds")
        print(f"\n------------\n")

        # saving in file
        with open("./time_experiment_reults.txt", "a") as file:
            file.write(results)
    

def revert_sim(sim:float):
    return round((sim*-1)+1.0, 2)


def main():
    # load data
    data = pd.read_excel("./data_scientist_united_states_job_postings_jobspikr.xlsx")
    choose_a_post = False

    # choose a post
    while not choose_a_post:
        post_id = get_number_input(f"Type a number between 0 and {data.shape[0]} to choose a job post", 0, data.shape[0]-1)
        
        post = data.values.tolist()[post_id]
        print_job_post(post)
        answer = get_bool_input("Is this ok? (y/n)")
        if answer == 1:
            choose_a_post = True

    # choose weights
    title_w = get_float_number_input(f"How important is the title? Type a number between 0.0 and 3.0 or pass/enter", 0.0, 3.0)
    title_w = 2.0 if title_w == None else title_w
    category_w = get_float_number_input(f"How important is the category? Type a number between 0.0 and 3.0 or pass/enter", 0.0, 3.0)
    category_w = 1.0 if category_w == None else category_w
    type_w = get_float_number_input(f"How important is the type? Type a number between 0.0 and 3.0 or pass/enter", 0.0, 3.0)
    type_w = 0.0 if type_w == None else type_w
    pos_w = get_float_number_input(f"How important is the location? Type a number between 0.0 and 3.0 or pass/enter", 0.0, 3.0)
    pos_w = 0.0 if pos_w == None else pos_w

    # calc scores -> get similar job posts
    start = time.time()
    #data.sample(n=2000, replace=False)
    posts = get_similar_job_posts(data.sample(n=1000, replace=False), post, title_w=title_w, category_w=category_w, \
                                            type_w=type_w, pos_w=pos_w, printing=False)
    duration = time.time() - start
    print(f"\n\nFounded similiar posts in {duration:.2f} seconds")


    # navigate in example
    cur_idx = 0
    print("-----\nNavigate with 'next', 'prev', 'exit'\n-----")
    while True:
        print_job_post(posts.iloc[cur_idx, :], False)
        user_input = input("User:")
        if user_input == "next" or user_input == "":
            if cur_idx < posts.shape[0]-1:
                cur_idx += 1
        elif user_input == "prev":
            if cur_idx > 0:
                cur_idx -= 1
        elif user_input == "exit":
            print("bye")
            break


if __name__ == "__main__":
    main()
    #time_experiment()

