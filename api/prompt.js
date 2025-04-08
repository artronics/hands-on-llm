// This file reads data from the data collection and construct a prompt for generative AI.
import soc from "data/soc_full_line";

function make_suitability_prompt(question, claimant, advert) {
    const preferredJobs = claimant.contentData.preferredJobs.yesNo
        ? `prefers to work on ${claimant.contentData.preferredJobs.values.join(", ")}`
        : "and has no specific preference";
    const hasJobExp = claimant.contentData.hasWorkHistory
        ? `has job experience and ${preferredJobs}`
        : "doesn't have job experience";


    const claimantQuery = `Following is information about the job seeker.\\nThe job seeker ${hasJobExp}`;

    const advertQuery = `Following is the job advert.\\nJob Title: ${advert.Title}\\nJob Description: ${advert.JobDescription}`;

    return `${question}\\n${claimantQuery}\\n${advertQuery}`;
}

function make_soc_labeling_prompt(advert) {
    return "here is a list of labels presented in json format. Each label defines a job role presented as"
        + "a progression of categories separated by pipe (|) character"
        + "I give you a job description and tell me which item in this array fits best. Only return the item and nothing else."
        + `labels:\\n${JSON.stringify(soc).replaceAll("\"", "'")}`
        + `job description:\\n${advert.JobDescription}`
}

export {make_suitability_prompt, make_soc_labeling_prompt};