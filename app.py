import streamlit as st
import tensorflow as tf
import numpy as np

# ==========================================
# Dataset
# ==========================================

data = [
    ("i am a student", "నేను విద్యార్థిని"),
    ("how are you", "మీరు ఎలా ఉన్నారు"),
    ("i love machine learning", "నాకు మెషిన్ లెర్నింగ్ అంటే ఇష్టం"),
    ("good morning", "శుభోదయం"),
    ("thank you", "ధన్యవాదాలు"),
    ("see you later", "తర్వాత కలుద్దాం"),
    ("what is your name", "మీ పేరు ఏమిటి"),
    ("where are you going", "మీరు ఎక్కడికి వెళ్తున్నారు"),
    ("i like coffee", "నాకు కాఫీ ఇష్టం"),
    ("welcome", "స్వాగతం")
]

english_sentences = [x[0] for x in data]

telugu_sentences = [
    "start " + x[1] + " end"
    for x in data
]

vocab_size = 1000
sequence_length = 20

# ==========================================
# Tokenization
# ==========================================

source_vectorization = tf.keras.layers.TextVectorization(
    max_tokens=vocab_size,
    output_mode="int",
    output_sequence_length=sequence_length
)

target_vectorization = tf.keras.layers.TextVectorization(
    max_tokens=vocab_size,
    output_mode="int",
    output_sequence_length=sequence_length,
    standardize=None
)

source_vectorization.adapt(english_sentences)
target_vectorization.adapt(telugu_sentences)

encoder_inputs = source_vectorization(english_sentences)
target_tokens = target_vectorization(telugu_sentences)

decoder_inputs = target_tokens[:, :-1]
decoder_targets = target_tokens[:, 1:]

# ==========================================
# Positional Embedding
# ==========================================

class PositionalEmbedding(tf.keras.layers.Layer):

    def __init__(self, sequence_length, vocab_size, embed_dim):
        super().__init__()

        self.token_embedding = tf.keras.layers.Embedding(
            vocab_size,
            embed_dim
        )

        self.position_embedding = tf.keras.layers.Embedding(
            sequence_length,
            embed_dim
        )

    def call(self, inputs):

        length = tf.shape(inputs)[-1]

        positions = tf.range(
            start=0,
            limit=length,
            delta=1
        )

        return (
            self.token_embedding(inputs)
            + self.position_embedding(positions)
        )

# ==========================================
# Encoder
# ==========================================

class TransformerEncoder(tf.keras.layers.Layer):

    def __init__(self, embed_dim, dense_dim, num_heads):
        super().__init__()

        self.attention = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim
        )

        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(dense_dim, activation="relu"),
            tf.keras.layers.Dense(embed_dim)
        ])

        self.norm1 = tf.keras.layers.LayerNormalization()
        self.norm2 = tf.keras.layers.LayerNormalization()

    def call(self, inputs):

        attention = self.attention(inputs, inputs)

        x = self.norm1(inputs + attention)

        ffn_output = self.ffn(x)

        return self.norm2(x + ffn_output)

# ==========================================
# Decoder
# ==========================================

class TransformerDecoder(tf.keras.layers.Layer):

    def __init__(self, embed_dim, dense_dim, num_heads):
        super().__init__()

        self.self_attention = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim
        )

        self.cross_attention = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim
        )

        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(dense_dim, activation="relu"),
            tf.keras.layers.Dense(embed_dim)
        ])

        self.norm1 = tf.keras.layers.LayerNormalization()
        self.norm2 = tf.keras.layers.LayerNormalization()
        self.norm3 = tf.keras.layers.LayerNormalization()

    def call(self, inputs, encoder_outputs):

        attention1 = self.self_attention(
            query=inputs,
            value=inputs,
            key=inputs,
            use_causal_mask=True
        )

        out1 = self.norm1(inputs + attention1)

        attention2 = self.cross_attention(
            query=out1,
            value=encoder_outputs,
            key=encoder_outputs
        )

        out2 = self.norm2(out1 + attention2)

        ffn_output = self.ffn(out2)

        return self.norm3(out2 + ffn_output)

# ==========================================
# Build and Train Model
# ==========================================

@st.cache_resource
def train_model():

    embed_dim = 128
    dense_dim = 256
    num_heads = 4

    encoder_input = tf.keras.Input(
        shape=(None,),
        dtype="int64"
    )

    x = PositionalEmbedding(
        sequence_length,
        vocab_size,
        embed_dim
    )(encoder_input)

    encoder_output = TransformerEncoder(
        embed_dim,
        dense_dim,
        num_heads
    )(x)

    decoder_input = tf.keras.Input(
        shape=(None,),
        dtype="int64"
    )

    x = PositionalEmbedding(
        sequence_length,
        vocab_size,
        embed_dim
    )(decoder_input)

    x = TransformerDecoder(
        embed_dim,
        dense_dim,
        num_heads
    )(x, encoder_output)

    decoder_output = tf.keras.layers.Dense(
        vocab_size,
        activation="softmax"
    )(x)

    model = tf.keras.Model(
        [encoder_input, decoder_input],
        decoder_output
    )

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.fit(
        [encoder_inputs, decoder_inputs],
        decoder_targets,
        epochs=100,
        batch_size=2,
        verbose=0
    )

    return model

transformer = train_model()

# ==========================================
# Translation Function
# ==========================================

def translate(sentence):

    encoder_input_test = source_vectorization([sentence])

    decoded_sentence = "start"

    vocab = target_vectorization.get_vocabulary()

    index_lookup = dict(
        zip(range(len(vocab)), vocab)
    )

    for i in range(sequence_length - 1):

        tokenized_target = target_vectorization(
            [decoded_sentence]
        )

        predictions = transformer.predict(
            [encoder_input_test, tokenized_target],
            verbose=0
        )

        token_index = np.argmax(
            predictions[0, i, :]
        )

        token = index_lookup.get(token_index, "")

        if token == "end":
            break

        decoded_sentence += " " + token

    return decoded_sentence.replace(
        "start", ""
    ).strip()

# ==========================================
# Streamlit UI
# ==========================================

st.set_page_config(
    page_title="English to Telugu Translator",
    page_icon="🌐"
)

st.title("🌐 English → Telugu Translator")
st.write("Small Transformer-based translation demo.")

user_input = st.text_input(
    "Enter English Sentence:"
)

if st.button("Translate"):

    if user_input.strip() == "":
        st.warning("Please enter a sentence.")
    else:
        result = translate(user_input.lower())

        st.success("Translation")
        st.write(result)