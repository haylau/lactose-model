from wordcloud import WordCloud
import matplotlib.pyplot as plt

from train import freq_tokens

wordcloud = WordCloud(width=800, height=400, background_color="white").generate_from_frequencies(freq_tokens)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()
