from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

def calculate_bleu_from_text(reference_texts, candidate_text):
    """
    Calculate BLEU score for a single candidate text against one or more reference texts, with raw text input.
    
    :param reference_texts: A list of reference text strings.
    :param candidate_text: The candidate text as a string.
    :return: BLEU score
    """
    # Tokenize the reference texts and the candidate text
    tokenized_references = [word_tokenize(ref) for ref in reference_texts]
    tokenized_candidate = word_tokenize(candidate_text)
    
    smooth = SmoothingFunction().method1
    return sentence_bleu(tokenized_references, tokenized_candidate, smoothing_function=smooth)

def calculate_wer_from_text(reference, hypothesis):
    """
    Calculate Word Error Rate (WER) between a reference text and a hypothesis text, with raw text input.
    
    :param reference: The reference text as a string.
    :param hypothesis: The hypothesis (candidate) text as a string.
    :return: WER
    """
    # Tokenize the reference and hypothesis texts
    reference_tokens = word_tokenize(reference)
    hypothesis_tokens = word_tokenize(hypothesis)
    
    # Initialize matrix
    d = [[0 for _ in range(len(hypothesis_tokens)+1)] for _ in range(len(reference_tokens)+1)]
    for i in range(len(reference_tokens)+1):
        for j in range(len(hypothesis_tokens)+1):
            if i == 0:
                d[i][j] = j
            elif j == 0:
                d[i][j] = i

    # Populate matrix
    for i in range(1, len(reference_tokens)+1):
        for j in range(1, len(hypothesis_tokens)+1):
            if reference_tokens[i-1] == hypothesis_tokens[j-1]:
                cost = 0
            else:
                cost = 1
            d[i][j] = min(d[i-1][j] + 1,        # deletion
                         d[i][j-1] + 1,        # insertion
                         d[i-1][j-1] + cost)   # substitution

    return d[len(reference_tokens)][len(hypothesis_tokens)] / float(len(reference_tokens))
