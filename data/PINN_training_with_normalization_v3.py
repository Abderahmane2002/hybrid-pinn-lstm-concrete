
# ✅ Bloc complet avec normalisation, softplus, affichage D0/loss

# 1. Définir la normalisation
X_mean = tf.constant([0.13, 20.0, 0.45, 0.6], dtype=tf.float32)
X_std = tf.constant([0.02, 5.0, 0.05, 0.05], dtype=tf.float32)

def normalize(x):
    return (x - X_mean) / X_std

# 2. Redéfinir le modèle ANN avec softplus
def build_ann_model(input_dim=4):
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(32, activation='tanh'),
        Dense(32, activation='tanh'),
        Dense(1, activation='softplus')  # Toujours positif
    ])
    return model

model = build_ann_model()
optimizer = Adam(learning_rate=0.01)

# 3. Boucle d'entraînement
loss_history = []
input_tensor = tf.convert_to_tensor([[0.119, 20.0, 0.45, 0.6]], dtype=tf.float32)

for epoch in range(300):
    with tf.GradientTape() as tape:
        D0 = model(normalize(input_tensor))
        Dv = tf.squeeze(D0)
        x_model, w_model = simulate_model_tf(Dv, 0.6, 0.119, 1000.0, 1e-3, 100.0, 72.0*3600, 60)

        def interp_fn(w):
            return np.interp(x_exp.numpy(), x_model.numpy(), w.numpy())

        w_interp = tf.numpy_function(interp_fn, [w_model], tf.float32)
        loss = tf.reduce_mean(tf.square(w_interp - w_exp))

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    loss_history.append(loss.numpy())

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, D0: {D0.numpy()[0][0]:.2e}, Loss: {loss.numpy():.4f}")
