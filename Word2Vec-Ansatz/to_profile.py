import job_post_sim
import pandas as pd

data = pd.read_excel("../data_scientist_united_states_job_postings_jobspikr.xlsx")
post = data.values.tolist()[33]

job_post_sim.get_similar_job_posts(data.sample(n=1000, replace=False), post, title_w=2.0, category_w=1.0, \
                                            type_w=0.0, pos_w=0.0, printing=False)

