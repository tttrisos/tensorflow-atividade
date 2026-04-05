async function treinarEPrever() {

  // Elementos
  const textoStatus = document.getElementById("status");
  const textoResultado = document.getElementById("resultado");

  // Valor digitado
  const anosExperiencia = Number(
    document.getElementById("experiencia").value
  );

  textoStatus.innerText = "Status: Treinando a IA...";

  // 1. CRIAR O MODELO
  const modelo = tf.sequential();
  modelo.add(
    tf.layers.dense({
      units: 1,
      inputShape: [1],
    })
  );

  // 2. COMPILAR O MODELO
  modelo.compile({
    loss: "meanSquaredError",
    optimizer: "sgd",
  });

  // 3. DADOS DE TREINO
  // X = anos de experiência
  // Y = salário (em milhares)
  const dadosEntrada = tf.tensor2d(
    [1, 2, 3, 4, 5],
    [5, 1]
  );

  const dadosSaida = tf.tensor2d(
    [2000, 3000, 4000, 5000, 6000],
    [5, 1]
  );

  // 4. TREINAMENTO
  await modelo.fit(dadosEntrada, dadosSaida, {
    epochs: 200,
  });

  textoStatus.innerText = "Status: IA treinada!";

  // 5. PREVISÃO
  const previsao = modelo.predict(
    tf.tensor2d([anosExperiencia], [1, 1])
  );

  const salarioPrevisto = previsao.dataSync()[0];

  textoResultado.innerText =
    "Salário estimado: R$ " +
    salarioPrevisto.toFixed(2);
}