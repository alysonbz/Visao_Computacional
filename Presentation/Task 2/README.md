# Atividade

**Objetivo** 

Construir um mini dataset com 2 classes e 10 imagens no total, aplicando na prática os fundamentos discutidos em aula:

* Resolução e profundidade de cor
* Conversões de espaço de cor (RGB, HSV, escala de cinza)
* Aquisição (condições de captura)
* Armazenamento e formatos (RAW/JPEG/PNG)
* Análise de impacto visual e funcional das escolhas
* Resposta no google Colab em markdown um relatório em conjunto com os códigos respostas.

Esta atividade simula etapas reais de um projeto profissional:

* Construção de dataset próprio.
* Controle de aquisição.
* Decisão sobre formato e resolução.
* Análise de impacto técnico antes de treinar modelos.

Em aplicações industriais, médicas ou científicas, essas decisões definem o sucesso ou fracasso do sistema. O objetivo é que o aluno desenvolva mentalidade de engenharia, não apenas execução técnica.

**Enunciado:**

Você deverá capturar 10 imagens originais (feitas pela equipe, com câmera do celular), que possam ser separadas em duas classes (5 imagens por classe). Em seguida, você deverá aplicar transformações e decisões de armazenamento com base nos fundamentos de Processamento de Imagens Digitais vistos em aula, organizando ao final um mini dataset estruturado, pronto para ser usado em tarefas de Visão Computacional.

**Instruções:**

**Parte 1: Definição das classes** 

Escolha um tema simples, com boa distinção visual e fácil coleta:
Sugestões:

* Classe A: “Folha” | Classe B: “Flor”
* Classe A: “Copo” | Classe B: “Garrafa”
* Classe A: “Teclado” | Classe B: “Mouse”
* Classe A: “Porta” | Classe B: “Janela”
* Classe A: “Objeto com cor dominante vermelha” | Classe B: “Objeto com cor dominante azul”

**obs: Você pode escolher outro exemplo de imagem, acima são sugestões opcionais.**

Regras:

* As duas classes devem ser coerentes e claramente definidas.
* As imagens precisam ser capturadas por você , não baixar da internet.

**Parte 2: Aquisição de imagens**

Capture 5 imagens por classe, totalizando 10.

Requisitos mínimos de captura:

* Pelo menos 2 cenários de iluminação no conjunto total, por exemplo com luz natural, sombra e  luz artificial.
* Pelo menos 2 distâncias/ângulos diferentes para cada classe.
* Evitar filtros automáticos, desativar embelezamento.

Você deverá registrar em texto em markdown:

* Dispositivo usado: smartphone modelo X
* Condição de iluminação
* Distância aproximada
* Observação de ruído, desfoque se houver.

**Parte 3: Transformações de Imagens**

Para 1 exemplo de cada classe, você deverá gerar versões derivadas.

1. **Análise de Resolução:**
    * 1 versão em resolução original
    * 1 versão com redução em 50% da resolução
    * 1 versão com redução em 20% da resolução

obs: Realize comentário no colab sobre perda de detalhe percebida e possível impacto em visão computacional

2. **Espaço de cor**

* Gerar 1 exemplo de cada classe visualização em:

    * RGB 
    * HSV.
    * Escala de cinza

Registro textual:

Quais diferenças visuais aparecem ao converter ?

Em que tipo de tarefa você usaria cada uma?

3. **Quantização**
Escolha uma imagem de cada classe e gere versões em:

* 256 níveis de cinza
* 64 níveis de cinza
* 32 níveis de cinza
* 2 níveis de cinza

Registrar:

O que muda visualmente?

Qual versão ainda seria mais útil para o seu dataset?  e para qual aplicação ?

4. **Formato de arquivo**
Salvar versões em formatos diferentes:

* JPEG com compressão
* PNG sem perdas

Registrar:

* Tamanho do arquivo (KB/MB)
* Diferença visual percebida
* Hipótese sobre impacto em tarefas de PDI

**Parte 4: Estrutura do mini dataset**

Você deverá organizar o dataset em pastas com padrão de ML:

Estrutura sugerida:

    dataset/

    ├── classe_A/
    │   ├── img001_original.jpg
    │   ├── img002_original.jpg
    │   ├── img003_original.jpg
    │   ├── img004_original.jpg
    │   └── img005_original.jpg
    └── classe_B/
        ├── img006_original.jpg
        ├── img007_original.jpg
        ├── img008_original.jpg
        ├── img009_original.jpg
        └── img010_original.jpg
