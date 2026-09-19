"""Frozen question templates. The teacher never writes the question text."""

from __future__ import annotations

from schema import Question

SUPPORT_BUNDLE: list[Question] = [
    Question(
        id="dept",
        type="choice",
        instructions="Which team should handle this customer request?",
        criteria={
            "billing": "Charges, invoices, refunds, payments, or account balance",
            "technical": "Bugs, outages, login, app, or product not working",
            "sales": "Buying, upgrading, pricing, or a new plan",
            "other": "Anything else, including unclear requests",
        },
    ),
    Question(
        id="tone",
        type="choice",
        instructions="What is the channel tone of this message?",
        criteria={
            "request": "The writer wants something done",
            "complaint": "The writer is unhappy with service or a product",
            "question": "The writer is asking for information",
            "praise": "The writer is thanking or complimenting",
        },
    ),
    Question(
        id="urgency",
        type="score",
        instructions="How urgent is this request?",
        criteria=[
            "Low. Can wait days. No deadline or harm.",
            "Medium. Should be handled this week.",
            "High. Immediate impact, outage, or explicit deadline.",
        ],
    ),
    Question(
        id="frustration",
        type="score",
        instructions="How frustrated is the writer?",
        criteria=[
            "Calm. Neutral or polite.",
            "Annoyed. Some irritation, still cooperative.",
            "Angry. Hostile, threats, or heavy blame.",
        ],
    ),
    Question(
        id="refund",
        type="noul",
        instructions="The customer asks for a refund.",
    ),
    Question(
        id="bug",
        type="noul",
        instructions="The message reports a bug or an outage.",
    ),
    Question(
        id="cancel",
        type="noul",
        instructions="The customer threatens to cancel.",
    ),
    Question(
        id="phish",
        type="noul",
        instructions="The message looks like phishing or social engineering.",
    ),
]

EMAIL_BUNDLE: list[Question] = [
    Question(
        id="type",
        type="choice",
        instructions="What type of email is this?",
        criteria={
            "transactional": "Receipt, password reset, shipping, account notice",
            "newsletter": "Marketing blast or digest",
            "personal": "One-to-one human correspondence",
            "spam": "Unsolicited junk, not targeted fraud",
            "phishing": "Tries to steal credentials or money",
        },
    ),
    Question(
        id="cta",
        type="noul",
        instructions="The message contains a call to action.",
    ),
    Question(
        id="credentials",
        type="noul",
        instructions="The message requests money or credentials.",
    ),
    Question(
        id="professionalism",
        type="score",
        instructions="How professional is the writing?",
        criteria=[
            "Unprofessional. Broken language, shouting, or scam style.",
            "Mixed. Informal but readable.",
            "Professional. Clear, grammatical, businesslike.",
        ],
    ),
]

BUNDLE_BY_DOMAIN = {
    "support": SUPPORT_BUNDLE,
    "email": EMAIL_BUNDLE,
}

# Three human-written paraphrases per gold instruction template. Not model-written.
PARAPHRASES: dict[str, list[str]] = {
    "banking77.intent": [
        "Which banking intent matches this customer message?",
        "What does the customer want the bank to do?",
        "Classify the banking request in this message.",
    ],
    "banking77.coarse": [
        "Which department-style bucket fits this banking message?",
        "Route this ticket to a coarse banking category.",
        "Pick the high-level banking topic.",
    ],
    "clinc.intent": [
        "Which intent best matches this utterance? Include out-of-scope if none apply.",
        "What is the user trying to do? Use out-of-scope when it is none of the listed intents.",
        "Classify this utterance into an intent, or out-of-scope.",
    ],
    "massive.intent": [
        "Which spoken-style intent matches this utterance?",
        "What does the speaker want?",
        "Classify the intent of this short spoken request.",
    ],
    "ag_news.topic": [
        "Which news topic is this article?",
        "Pick the topic bucket for this news text.",
        "Classify the article into a news section.",
    ],
    "trec.coarse": [
        "What type of answer does this question expect?",
        "Classify the question by the kind of thing it asks for.",
        "Which coarse question type is this?",
    ],
    "emotion.label": [
        "Which emotion does this text express?",
        "Pick the emotion label for this sentence.",
        "Classify the writer's emotion.",
    ],
    "bitext.intent": [
        "Which customer-support intent matches this chat?",
        "What does the customer want support to do?",
        "Classify the support intent.",
    ],
    "tweet.emotion": [
        "Which emotion does this tweet express?",
        "Classify the tweet's emotion.",
        "Pick the emotion label for this social post.",
    ],
    "tweet.offensive": [
        "Is this tweet offensive or not?",
        "Classify whether the tweet is offensive.",
        "Pick offensive or not-offensive for this tweet.",
    ],
    "boolq.yes": [
        "The passage answers the question with yes.",
        "According to the passage, the answer to the question is yes.",
        "The passage supports a yes answer to the question.",
    ],
    "rte.entail": [
        "The hypothesis follows from the premise.",
        "The premise entails the hypothesis.",
        "If the premise is true, the hypothesis must be true.",
    ],
    "qnli.contains": [
        "The sentence contains the answer to the question.",
        "This sentence answers the question.",
        "The question can be answered from this sentence.",
    ],
    "paws.paraphrase": [
        "The two sentences are paraphrases.",
        "The sentences mean the same thing.",
        "These two sentences are semantically equivalent.",
    ],
    "scitail.entail": [
        "The hypothesis is entailed by the premise.",
        "The scientific premise entails the hypothesis.",
        "The hypothesis follows from the premise.",
    ],
    "enron.spam": [
        "This email is spam.",
        "The message is unsolicited spam.",
        "Classify this email as spam.",
    ],
    "sms.spam": [
        "This message is spam.",
        "The SMS is unsolicited spam.",
        "This text message is spam.",
    ],
    "civil.toxic": [
        "This comment is toxic.",
        "The comment is toxic or abusive.",
        "The text contains toxic language.",
    ],
    "injection.inject": [
        "This prompt tries to inject or override instructions.",
        "The text is a prompt-injection attempt.",
        "The prompt tries to override the system's instructions.",
    ],
    "jailbreak.jailbreak": [
        "This prompt is a jailbreak attempt.",
        "The prompt tries to jailbreak the model.",
        "This is a jailbreak prompt.",
    ],
    "imdb.positive": [
        "The review is positive.",
        "The writer likes the film.",
        "This is a positive review.",
    ],
    "sst5.sentiment": [
        "How positive is this sentence on a five-level sentiment rubric?",
        "Rate the sentiment of this sentence from very negative to very positive.",
        "Pick the sentiment level of this sentence.",
    ],
    "yelp.stars": [
        "How many stars does this review correspond to?",
        "Rate the review on a one-to-five star rubric.",
        "Pick the star rating expressed by this review.",
    ],
    "amazon.stars": [
        "How many stars does this product review correspond to?",
        "Rate the review on a one-to-five star rubric.",
        "Pick the star rating expressed by this review.",
    ],
    "helpsteer.helpfulness": [
        "How helpful is this assistant response?",
        "Rate helpfulness of the response.",
        "Score how helpful the response is.",
    ],
    "helpsteer.correctness": [
        "How correct is this assistant response?",
        "Rate factual correctness of the response.",
        "Score how correct the response is.",
    ],
    "helpsteer.coherence": [
        "How coherent is this assistant response?",
        "Rate coherence of the response.",
        "Score how coherent the response is.",
    ],
    "helpsteer.complexity": [
        "How complex is this assistant response?",
        "Rate the complexity of the response.",
        "Score how complex the writing is.",
    ],
    "helpsteer.verbosity": [
        "How verbose is this assistant response?",
        "Rate verbosity of the response.",
        "Score how verbose the response is.",
    ],
    "civil.toxicity_level": [
        "How toxic is this comment?",
        "Rate the toxicity of the comment.",
        "Pick the toxicity level of this comment.",
    ],
    "app.stars": [
        "How many stars does this app review correspond to?",
        "Rate the app review on a one-to-five star rubric.",
        "Pick the star rating expressed by this review.",
    ],
    "support.dept": [
        "Which team should handle this customer request?",
        "Route this ticket to a department.",
        "Who owns this request?",
    ],
    "support.tone": [
        "What is the channel tone of this message?",
        "Is this a request, complaint, question, or praise?",
        "Classify the tone of the customer message.",
    ],
    "email.type": [
        "What type of email is this?",
        "Classify the email as transactional, newsletter, personal, spam, or phishing.",
        "Pick the email type.",
    ],
}


STAR_CRITERIA = [
    "One star. The reviewer is fully negative.",
    "Two stars. Major complaints, little praise.",
    "Three stars. Mixed, neither strong praise nor strong attack.",
    "Four stars. Mostly positive with a small complaint.",
    "Five stars. Fully positive.",
]

SST5_CRITERIA = [
    "Very negative.",
    "Negative.",
    "Neutral.",
    "Positive.",
    "Very positive.",
]

HELPSTEER_CRITERIA = [
    "0. Not at all.",
    "1. Slightly.",
    "2. Moderately.",
    "3. Mostly.",
    "4. Fully.",
]

TOXICITY_ORDINAL = [
    "None. Not toxic.",
    "Mild. Some rude or heated language.",
    "Strong. Clearly toxic or insulting.",
    "Severe. Hateful, threatening, or extremely abusive.",
]
