Resumo geral do projeto
Sistema de Detecção Automática de Defeitos em Peças Metálicas Industriais

Este projeto propõe o desenvolvimento de um sistema de visão computacional capaz de detectar automaticamente defeitos em peças metálicas produzidas em uma linha industrial, como discos de freio automotivos.

O objetivo do sistema é automatizar o processo de inspeção de qualidade, identificando defeitos como:

rachaduras

oxidação

riscos

irregularidades na superfície

A utilização de visão computacional permite aumentar a precisão da inspeção, reduzir erros humanos e acelerar o processo de controle de qualidade.

Aquisição e representação das imagens

As imagens das peças seriam capturadas por câmeras industriais posicionadas na linha de produção, sob condições controladas de iluminação para garantir maior consistência nos dados.

Para o sistema proposto foram definidas as seguintes características de imagem:

Resolução espacial: 512 × 512 pixels

Espaço de cor: escala de cinza (grayscale)

Profundidade de cor: 8 bits por pixel

Essa configuração permite capturar detalhes da textura da superfície metálica, que são fundamentais para a identificação de defeitos.

Processamento e análise das imagens

O sistema analisa principalmente:

intensidade dos pixels

padrões de textura

variações locais de brilho

Esses atributos permitem identificar alterações na superfície da peça, que podem indicar defeitos estruturais ou problemas no processo de fabricação.

Considerações sobre custo computacional

Cada imagem possui aproximadamente:

262.144 pixels

tamanho médio de 256 KB

Considerando uma produção de aproximadamente 5.000 peças por dia, o sistema geraria cerca de:

1,28 GB de dados por dia

O processamento estimado é de aproximadamente bilhões de operações por dia, o que pode ser executado em computadores convencionais, tornando o sistema viável para aplicação industrial.

Conclusão

O sistema proposto demonstra como técnicas de processamento digital de imagens e visão computacional podem ser aplicadas na indústria para melhorar o controle de qualidade.

A escolha de imagens em escala de cinza e resolução moderada permite equilibrar qualidade de detecção e custo computacional, tornando a solução eficiente e aplicável em ambientes industriais.