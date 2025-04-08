## Using API
Create a new json file `http-client.private.env.json` with the below content.

```json
{
  "dev": {
    "apikey": "<replace-this-with-your-openai-api-key>"
  }
}
```
The `openai.http` has request examples for different use cases. For example below pre-request script creates this prompt:
```js
    const claimant = claimants[0];
    const advert = adverts[0];
    const question = "Tell me if this job seeker is suitable for the job."

    const prompt = make_suitability_prompt(question, claimant, advert);

```
```text
Tell me if this job seeker is suitable for the job.\nFollowing is information about the job seeker.\nThe job seeker has job experience and prefers to work on Reception, Admin\nFollowing is the job advert.\nJob Title: Java Developer\nJob Description: We need Java Developer!

```

## Using openai library
Most of openai examples uses their library. The `openai.js` file uses an example. To run this script you need to export 
your apikey: `export OPENAI_API_KEY="your_api_key_here"` and then `node openai.js`. Or, if you already have the `http-client.private.env.json` file as mentioned before
then you can just run `npm run openai`