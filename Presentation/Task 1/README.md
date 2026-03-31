# Tarefa - 16-03

**Objetivo**

Desenvolver uma análise teórica  sobre os fundamentos estruturais das imagens digitais, articulando:

* Estrutura matricial da imagem
* Pixel, resolução espacial e radiométrica
* Espaços de cor (RGB, HSV e escala de cinza)
* Pipeline de aquisição da imagem
* Formatos de armazenamento e compressão
* Impacto dessas decisões em sistemas de Visão Computacional

O foco aqui é a argumentação técnica e conceitual.

 

**Enunciado:**

Em projetos reais de Visão Computacional, as decisões tomadas antes do treinamento de qualquer modelo, como resolução escolhida, espaço de cor utilizado ou formato de armazenamento,  influenciam diretamente o desempenho, custo computacional e robustez do sistema.

Seu papel nesta atividade será assumir a posição de um(a) engenheiro(a) responsável por projetar um sistema de visão computacional para um dos cenários abaixo. Esta tarefa busca expandir de forma estruturada o que foi estudado até agora. Para se dar bem nela é preciso não só estudar o que está contido aqui no curso, mas também buscar por fontes externas, como artigos, blogs, vídeos e tudo aquilo que você está acostumado a utilizar nos seus estudos.

Pense que quando você estiver trabalhando, é assim que você irá desenvolver seus projetos. Muitas vezes, você terá o conhecimento técnico, mas precisará correr atrás das questões relacionadas ao domínio de aplicação, ou seja, o segmento específico do mundo real para o qual seu sistema de visão computacional será desenvolvido.

 

**Instruções:**

Escolha um dos cenários propostos:

* Sistema de detecção automática de defeitos em peças metálicas industriais
* Sistema de triagem médica por imagem 
* Sistema de reconhecimento de frutas em um supermercado automatizado
* Sistema de leitura automática de placas veiculares
* Sistema de classificação de resíduos recicláveis
* Um cenário diferente do seu interesse

Desenvolva todo o contexto do cenário escolhido (tamanho da empresa, segmento, etc) e produza uma apresentação em PPT respondendo às questões abaixo de forma argumentativa e fundamentada no conteúdo visto em aula e no conteúdo extra que você pesquisará, como exemplo, este artigo discute várias aplicacões em diversos setores da indústria: https://repositorio.usp.br/directbitstream/e9aced0b-d8da-4956-b2e0-bd3bcbeceb24/Vis%C3%A3o%20comput… .

 Na sua apresentação, responda perguntas fundamentais como:

* Qual é a relevância da resolução espacial para o seu problema? É preciso uma resolução alta ou não?
* Qual é a relevância da profundidade de cor para o seu problema? É preciso uma resolução alta ou não?
* Ao tratar os dados do seu problema, o espaço de cor,  tem influência? Qual seria escolhido e porquê?
* Para o seu problema, é mais importante o detalhamento individual de cada pixel ou ele aceita perda de informação em troca de arquivos de dados menores e assim, maior rapidez no processamento?
* Você teria controle sobre o processo de captura de dados para seu sistema? Como montaria este processo?
* Se você tivesse orçamento limitado, qual decisão técnica sacrificaria primeiro: resolução, profundidade de cor, processo de captura?  
 

OBS: A apresentação deve ser inclusa na pasta "presentation" do repositório do github com o nome de tarefa 1


