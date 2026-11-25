 - Panografia (Costura de Imagens Transladadas)
Descrição:
🎯 Objetivo
Implementar, na casca de programa solicitada anteriormente, uma aplicação simples de panografia, em que duas imagens parcialmente sobrepostas são alinhadas por translação e mescladas em um mosaico contínuo, formando uma única imagem panorâmica.

🧠 Descrição
Nesta atividade, você implementará uma versão simplificada do processo de costura de imagens (image stitching), utilizando apenas o modelo de transformação por translação.
O objetivo é alinhar as imagens de forma que as características correspondentes coincidam no mesmo quadro global, produzindo um mosaico panorâmico suave.

📦 Entradas
Duas imagens fornecidas (img1.jpg e img2.jpg), com sobreposição parcial e deslocamento por translação.

📤 Saídas Esperadas
Imagem panorâmica resultante (mosaico.jpg);

Vetor de translação estimado t = (tx, ty);

Quantidade de correspondências válidas (inliers) encontradas;

Comparação visual entre o mosaico e a justaposição direta das imagens.

⚙️ Tarefas
Detectar características nas duas imagens (ex.: SIFT, ORB, Harris, etc.).

Encontrar correspondências entre as imagens.

Estimar a translação entre as imagens a partir das correspondências encontradas.

Pode-se usar uma abordagem robusta, como RANSAC, para eliminar outliers.

Imprimir os resultados estimados:

Vetor de translação t = (tx, ty);

Número de correspondências válidas (inliers).

Aplicar a transformação (translação) na segunda imagem para alinhá-la à primeira.

Gerar o mosaico final, combinando as duas imagens — por média simples ou blending linear na região de sobreposição.

Exibir e comparar:

O resultado da panografia (mosaico).

A justaposição direta das imagens sem alinhamento (colocadas lado a lado para referência).