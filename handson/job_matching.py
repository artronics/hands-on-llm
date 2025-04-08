import json

import kagglehub
from sentence_transformers import SentenceTransformer, InputExample, losses
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from datasets import load_dataset
from torch.utils.data import DataLoader

import pandas as pd


def make_example_entry(df) -> InputExample:
    title = df["Title"]

    jd = df["JobDescription"]
    jr = df["JobRequirment"]  # don't fix the spelling. it's in the data
    text = f"{jd}\n{jr}"


    label = 1.0 if df['Title'] else 0.0  # Match = 1.0, No Match = 0.0

    return InputExample(texts=[text, title], label=label)


def make_example(df: pd.DataFrame) -> [InputExample]:
    examples = []
    df["Title"] = df["Title"].fillna("").astype(str)
    df["JobDescription"] = df["JobDescription"].fillna("").astype(str)
    df["JobRequirment"] = df["JobRequirment"].fillna("").astype(str)

    for _, row in df.iterrows():
        examples.append(make_example_entry(row))

    return examples


def fine_tune():
    path = kagglehub.dataset_download("udacity/armenian-online-job-postings")
    df = pd.read_csv(f"{path}/online-job-postings.csv")
    examples = make_example(df)[:4000]
    train_dataloader = DataLoader(examples, shuffle=True, batch_size=16)

    model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    train_loss = losses.CosineSimilarityLoss(model)

    # === Fine-tune the model ===
    num_epochs = 3
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=num_epochs,
        warmup_steps=100,
        show_progress_bar=True
    )
    model.save("fine_tuned_job_classifier")


def match(advert):
    model = SentenceTransformer("fine_tuned_job_classifier")
    with open('job_titles.json') as f:
        job_titles = json.load(f)
    jt_embeddings = model.encode(job_titles, batch_size=16)

    advert_embedding = model.encode([advert])

    similarities = cosine_similarity(advert_embedding, jt_embeddings).flatten()

    top_indices = np.argsort(similarities)[::-1]
    top_matches = [(job_titles[idx], similarities[idx]) for idx in top_indices]

    return top_matches

advert_1 = """
Roadside Rescue Mechanic
Full job description
Roadside Rescue Mechanic

£54,000 OTE

Ready to be Always Ahead? So are we.

As one of our Roadside Rescue Mechanics you’ll be there for our customers, come rain or shine. Always ready to provide friendly help and reassurance, you’ll be more than a brilliant mechanic – you’ll be a genuine people person and ready to make a positive impact with everyone you meet.

What you’ll be doing:

· Your working day: You’re paid from the minute you get in your van to the moment you get home

· Work/Life balance: Choose the standby hours to suit your lifestyle

· Equipment: You bring your skills and expertise and we provide the rest, from a van and the very best tools to your uniform and boots

· Your team: You’ll join a tight-knit, supportive team and enjoy great development and training opportunities

· Our company: The AA is loved and recognised by all our customers

What you’ll need

NVQ3 in Vehicle Maintenance and Repair, or equivalent qualifications with appropriate experience
· A full category B driving licence, with 6 points or less.

· You should be happy to work shifts, which could include evenings, weekends and Bank Holidays

· You’ll be ready to work both independently and as part of a team, driving to different locations in all weathers

· A superb communicator, you’ll be skilled at explaining things to our customers so they’re reassured and know what’s going on

What’s in it for me?

· Free breakdown cover from day one

· 23 days holidays (increases with service) plus bank holidays

· Up to 7% company pension contribution

· Industry leading training

· Dedicated employee assistance programme and a 24/7 remote GP service for you and your family

· A welcoming, inclusive culture that will help you thrive

When you're with The AA, you're Always Ahead!

Interested? Apply today.

Job Types: Full-time, Permanent

Pay: £36,140.00-£54,000.00 per year

Benefits:

Company car
Company pension
On-site parking
Schedule:

Day shift
Flexitime
Monday to Friday
Night shift
Weekend availability
Work Location: On the road
"""
advert_2 = """
Title: Lead Engineer (Specialist)
About the job
Location: Leeds or London
The Bank has started gaining a greater foothold in cloud services over the last few years. As we grow, we are evolving our operating model to allow us to run our services more effectively. This is a lead role within the Application Cloud Services (ACS) platform team within DES. It is responsible for the day-to-day operations and technical backlog of the Azure platform where we host our applications including the Banks primary Content Management System, Sitecore!
The team will be responsible for monitoring elements of the Azure platforms pertinent to our applications and solving any issues which may fall from there. The team will have responsibility for running certain features in the platform, such as deploying golden images, handling subscriptions or upgrades of Kubernetes. They will also help the Microsoft Power Platform and Azure Data teams with platform support and build. 
This role on our the ACS team will be the Banks Sitecore Subject Matter Expert, and will act as the primary interface between the wider Azure Platform team and our 3rd party application support team. You will be responsible for managing the 'cloud platform' components of some solutions, providing strategic guidance and sharing knowledge within the team. You will work alongside the website team on our SiteCore CMS system and liaise with our external partners who support the website content. You will be challenged to find ways to automate your platform work and identifying and implementing continuous improvement. You will ensure the application platform has industry best practices in place for monitoring and alerting as well as managing logs. Ensuring professional management of patching, certificates and RBAC across the platform. You will use your SiteCore experience to ensure best practices are adhered to and that the Bank delivers the best and most robust system it can. 
We do not expect new joiners to have expert skills across all the technical areas we are looking towards, but we do want you to enjoy learning and will support you to fill technical gaps. We are looking towards our Microsoft and Oracle partnerships to support us with this education as this may include Microsoft Power Platform and Dynamics 365 as well as Oracle Cloud applications (Fusion).
Responsibilities include:
You will act as the Sitecore Product Owner for the Bank
You will take the lead in technical conversations and design activities to support the growth and maturity of the wider Sitecore Platform
You will work alongside key partners from the business ensuring we have a clear roadmap aligned to business requirements
You will work alongside colleagues in Technology and our 3rd party supplier to ensure work is planned and delivering to the same goals, and bring the work of the teams together 
You will champion continuous delivery in software engineering through use of Dev/Ops practices and your industry knowledge.
Role Requirements
Essential Criteria
Expert knowledge of the SiteCore platform
Experience of operating platforms in a controlled environment with strict change control.
Good knowledge of SOLR and Zookeeper
Experience in managing Kubernetes environments.
Experience with Azure monitoring, logging, and observability tools.
Preference to work as part of a team rather than individually.
Expert Microsoft Azure and CAF knowledge.
Microsoft Azure administration skills / certification.
Windows / Linux Admin.
Automation scripting (e.g. Terraform).
Strong written and verbal communication skills.
Be able to demonstrate a high level of professionalism, organisation, self-motivation and a desire for self-improvement.
Ability to plan, schedule and handle a demanding workload.
Desirable Criteria
Knowledge of Akamai CDN
Good .NET and C# skills.
SSIS skills and data integration/migration software.
In-depth experience of solution delivery using the full Software Development Lifecycle. This should include experience of working with the Agile and DevOps methodologies working with continuous delivery, automatic regression testing, TDD and monitoring.
Good knowledge of Azure Networks.
Experience of Microsoft Power Platform and Dynamics 365 environments.
 
Our Approach to Inclusion
The Bank values diversity, equity and inclusion. We play a key role in maintaining monetary and financial stability, and to do that effectively, we believe we need a workforce that reflects the society we serve. 
 
At the Bank of England, we want all colleagues to feel valued and respected, so we're working hard to build an inclusive culture which supports people from all backgrounds and communities to be at their best at work. We celebrate all forms of diversity, including (but not limited to) age, disability, ethnicity, gender, gender identity, race, religion, sexual orientation and socioeconomic status. We believe that it's by drawing on different perspectives and experiences that we'll continue to make the best decisions for the public. 
 
We welcome applications from individuals who work flexibly, including job shares and part time working patterns. We've also partnered with external organisations to support us in making adjustments for candidates and employees in the recruitment process where they're needed. 
 
For most roles where work can be carried out at home, we aim for colleagues to spend half of their time in the office, with a minimum of 40% per month. Subject to that minimum requirement, individuals and managers should work together to find what works best for them, their team and stakeholders. 
 
Finally, we're proud to be a member of the Disability Confident Scheme. If you wish to apply under this scheme, you should check the box in the 'Candidate Personal Information' under the 'Disability Confident Scheme' section of the application.
Salary and Benefits Information
We encourage flexible working, part time working and job share arrangements. Part time salary and benefits will be on a pro-rated basis as appropriate.
We offer a salary as follows:
Leeds circa £72,320 - £81,360
London circa £80,320 - £90,360
Currently a non-contributory, career average pension giving you a guaranteed retirement benefit of 1/95th of your annual salary for every year worked. There is the option to increase your pension (to 1/50th) or decrease (to 1/120th) in exchange for salary through our flexible benefits programme each year. The Bank has the discretion to vary standard accrual rates and dial up and dial down rates at any time and to withdraw dial up and dial down options at any time.
A discretionary performance award based on a current award pool.
A 8% benefits allowance with the option to take as salary or purchase a wide range of flexible benefits. 
26 days' annual leave with option to buy up to 12 additional days through flexible benefits. 
Private medical insurance and income protection.

National Security Vetting Process
Employment in this role will be subject to the National Security Vetting clearance process (and typically can take between 6 to 12 weeks post offer) and the passing of additional Bank security checks in accordance with the Bank policy. Further information regarding the vetting and security clearance requirements for the role will be provided to the successful applicant, and information about how the Bank processes personal data for these purposes, is set out in the Bank's Privacy Notice.
 
The Bank of England welcomes applications from all candidates, but as a UK Visas and Immigration (UKVI) approved sponsor, we have a responsibility to comply with the Immigration Rules and guidance. As such, our ability to employ individuals who require sponsorship for immigration purposes is limited. The Bank cannot guarantee that you and / or the role you are applying for will be eligible for sponsorship and that any application made to UKVI will be successful. Eligibility will therefore be considered on a case by case basis.
 
The Application Process
Important: Please ensure that you complete the 'work history' section and answer ALL the application questions fully. All candidate applications are anonymised to ensure that our hiring managers will not be able to see your personal information, including your CV, when reviewing your application details at the screening stage. It's therefore really important that you fill out the work history and application form questions, as your answers will form a critical part of the initial selection process. 
 
The assessment process will comprise of two interview stages. 

 
This role closes on 21 March 2025. 
 
Please apply online, ensuring that you complete your work history and answer ALL the application questions fully and in detail as your application will not be considered if all mandatory questions are not fully completed.



Desired Skills and Experience

Role Requirements

Essential Criteria

Expert knowledge of the SiteCore platform
Experience of operating platforms in a controlled environment with strict change control.
Good knowledge of SOLR and Zookeeper
Experience in managing Kubernetes environments.
Experience with Azure monitoring, logging, and observability tools.
Preference to work as part of a team rather than individually.
Expert Microsoft Azure and CAF knowledge.
Microsoft Azure administration skills / certification.
Windows / Linux Admin.
Automation scripting (e.g. Terraform).
Strong written and verbal communication skills.
Be able to demonstrate a high level of professionalism, organisation, self-motivation and a desire for self-improvement.
Ability to plan, schedule and handle a demanding workload.
Desirable Criteria

Knowledge of Akamai CDN
Good .NET and C# skills.
SSIS skills and data integration/migration software.
In-depth experience of solution delivery using the full Software Development Lifecycle. This should include experience of working with the Agile and DevOps methodologies working with continuous delivery, automatic regression testing, TDD and monitoring.
Good knowledge of Azure Networks.
Experience of Microsoft Power Platform and Dynamics 365 environments.
"""
advert = """
Title: Senior Software Engineer (Java)
About the job
CreateFuture is fast becoming the UK’s most recognisable digital consultancy, with years of experience building digital products and services for major organisations whilst putting our people first. We have offices in the centre of Edinburgh, Leeds, Manchester, and London as well as remote employees located throughout the country.

We are a team of creators - whether that’s code, project plans, go to market strategies, culture initiatives, marketing campaigns, large language models or people policies. And together, with our clients, we create the future. This has seen us collaborate and partner across a multitude of industries and sectors, with the likes of PayPal, adidas, Natwest, FanDuel and Money Saving Expert, to name just a few.

Our reputation as a partner determined to deliver high-quality, robust and thoughtful products has enabled us to scale to over 500 people in the last couple of years, and it is our amazing people - along with the safe, supportive and friendly culture we have built - that makes CreateFuture a great place to work. Don’t just take our word for it though, we have been recognised by Best Workplaces UK multiple years in a row - across a number of categories - and our employee exit rate is astonishingly low.

Join us on our journey… Let’s create something awesome, together, today.



About the role and team:
You will form part of a cross-functional agile team, working closely with 3-5 other engineers working closely with QA, Product and Design teams to deliver high-quality work for clients across a number of industries.

We strive to create the best product possible with everyone's effort, and you'll be an integral part of that mission. Your role in the team is highly valued, and you'll be a key player in ensuring the success of our projects.

We define our Senior role as someone who is able to confidently champion technical improvements, sets the bar high for best engineering practices and takes responsibility for the overall success of the project for the client and CeateFuture.

Our Tech:
Our backend team focuses on the following tech stack:
Java
RESTful Java APIs
CI/CD
AWS

What you'll be doing:
Developing RESTful Java APIs, including interaction with databases with a focus on readable, maintainable and well-tested code
Integrating with third-party APIs such as payment gateways
Being an advocate for engineering best practice within your project team
Help to grow our engineering function by participating in initiatives designed to increase our technical capabilities
Collaborating with the full project team (including BAs, QAs and members of the client team) to provide the best solutions to our clients
Identifying bottlenecks in the software delivery process and removing any roadblocks that arise
Professional handling of difficult client communications for specific issues
Identifying knowledge gaps and mentoring opportunities for your team
Being able to define and oversee the technical implementation of a client project

We'd love to talk to you if: 
You have strong knowledge of API development using Java, including ORM storage
You actively facilitate and encourage knowledge sharing within the team and wider department
You advocate for best engineering practices such as code reviews, paired programming, and automated tests
You are familiar with cloud services and CI/CD (this is a bonus)


What we'll offer you:
At CreateFuture, we challenge ourselves to go beyond the obvious and we care deeply about our craft and customers. With us, you’ll have ambitious projects to sink your teeth into and plenty of opportunities to learn and grow. You’ll be part of our safe, supportive and friendly culture - that looks after you - and join our team of genuinely great people.

Our benefits include:
Total 35 days holiday (we have flexible bank holidays)
Comprehensive private medical insurance
Enhanced parental and adoption leave
Pension - matched up to 5%

View our complete list of benefits here.

As this is a hybrid role, we’re looking for people within a commuting distance of our Leeds office and who are flexible to travel to client sites and CreateFuture regional offices. We are very flexible and trust you to manage your own schedule to balance face-to-face time with clients, colleagues and working from home.
We create and reinforce a culture that rewards employees’ impact, not just activity. We trust our employees to work autonomously and promote ownership across all levels.


Next steps
Our Talent Acquisition team aim to respond to all applications within a reasonable timeframe, regardless of if we are progressing with your application.

Our interview process:
30-minute recruiter call
1-hour technical assessment
Take home test or pair programming session + values interview


You can choose a short take-home task or a live pairing exercise for the final interview, whichever suits you best. Our interview process is designed as an opportunity both for our interviewers to learn about your expertise, interests and motivations and for you to gain insights into CreateFuture, the role and the team, so throughout the process, you’ll meet a few people from our backend team as well as others from our wider teams to help you get a well-rounded view of the role and life at CreateFuture.

We believe that representative teams made up of people with different backgrounds, skills, and points of view help us build the best workplace possible, and enable us to create genuinely innovative, broadly useful products.

We are committed to our goal of creating the most inclusive workplace possible. As we strive to build an environment where everyone can thrive and be themselves, we will continue to investigate and challenge biases, while working to identify and remove obstacles to inclusion. If you need additional support or accommodation during the application process, please don’t hesitate to let us know.
"""
def main():
    # fine_tune()
    matches = match(advert)
    print(matches[:10])


if __name__ == '__main__':
    main()
