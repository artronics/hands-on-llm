# Overview
## Challenges
* The biggest challenge for any AI related work is that we can't do it on our work-laptop. This is because both websites and services are blocked. So, even if wrote the code, we can't install its dependencies or execute it.
* Another challenge is anything related to fine-tuning a model. This process is very resource extensive. One solution can be to use cloud services such as AWS SageMaker for fine-tuning.
* There is also the problem of deployment. We don't know if we can deploy python but, even if we can, I think we should have a spike to explore the AWS solution, for both design and deployment.

## Representation Model
The content of the `handson` directory is all related to the exploring presentation model. As far as I learned so far, I think we can use presentation model to solve most of our problem. That's not say generative models aren't useful, but I think most our work relates to classification and semantic search which are better suited with presentation model. 

**Steps**:
These are the steps that involved to use a presentation model (see `docs/BERT` directory). I'm not sure if these are accurate or if there is any better way of doing it. This is based on my understanding so far doing tutorials and learning online.
* We need to choose a base model
  * Huggingface, is a repository of models and dataset and there are so many different models to choose from
    * There is a [benchmark tool](https://huggingface.co/spaces/mteb/leaderboard) that can be used to choose the right model. I didn't have luck in figuring out how it works!
  * There are models in there that are already fine-tuned for a specific task
    * [JobBERT-v2](https://huggingface.co/TechWolf/JobBERT-v2) for example. There are other models related to job
* We may need to fine-tune the model (optional)
  * This process involve choosing a dataset and create embedding based on a text (like job title) to another text (like job description) See `handson/job_matching.py`
  * It's resource extensive and, it takes a lot of time
  * We need a dataset for fine-tuning. There are a lot of them in both huggingface and kaggle. 
    * The challenging part is to clean the dataset before it can be used. 
  * We can use generative model to label the dataset. See the below section
* Now that we have sentence embeddings we can calculate the so-called cosine difference
  * There are other methods that is used to calculate the relatedness
  * `match(advert)` function uses this method to calculate the difference from a job advert to a list of job titles (soc in this case). Then we sort the result. The highest value (closer to 1.0) should be more related to the job advert.
        

## Generative Model
We can use generative model in two ways. One way is to use generative model to help us label our dataset, before using it for fine-tuning our presentation model. Another way of using it is to get insight, about data. OpenAI for example supports [structured output](https://platform.openai.com/docs/guides/structured-outputs) in response.
See the content of the `api` directory for examples of both labeling and insight. 
**NOTE**: I didn't try labeling for a large number of adverts but with the three examples that I tried it worked really well.
**NOTE**: I tried OpenAi `gpt-4o-mini` model. For labeling our dataset, the cost can be high. It would be good to use a few bigger and free models and run them locally and see if they perform as good.

### Can we use it for comparison between candidates?
Openai has a structured response mode but AFAIK there is no structure input. That doesn't mean we can't send json to it though. In fact, it recognises the json structure specially in `instruction` role. There are a few issues that we need to address:
* What quality is being measured here? In order to rank items in a list there must be one single overall metric so the list can be ordered
* How can we verify it? For example comparing between two items could be possible but between 100 candidates?
* How can we guarantee the model can count? We know for example, gen-ai can't keep track of items or do maths for example.
It may be possible to this using reasoning models or other methods but, I didn't research it.