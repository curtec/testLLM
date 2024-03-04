from transformers import pipeline

pipe = pipeline('ner')

answer = pipe('test Adam As part of your Queensland Membership in 2024 you will be able to redeem 2 tickets to Queensland matches where the Rabbitohs are the away team (excluding Magic Round). This can either be 2 tickets to 1 match or 1 ticket to 2 separate matches.')

print(answer)