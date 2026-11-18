const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, ImageRun, Table, TableRow, TableCell, WidthType, ShadingType,
  AlignmentType, HeadingLevel, LevelFormat, PageBreak, BorderStyle, TableOfContents, Footer, Header,
  PageNumber, TabStopType,
} = require("docx");

const ROOT = "/home/claude/reviewlens";
const FIG = ROOT + "/reports/figures/";
const M = JSON.parse(fs.readFileSync(ROOT + "/reports/metrics.json"));
const CV = JSON.parse(fs.readFileSync(ROOT + "/reports/cv_results.json"));
const Q = JSON.parse(fs.readFileSync(ROOT + "/reports/quality.json"));
const PREP = JSON.parse(fs.readFileSync(ROOT + "/reports/preparation.json"));
const CL = JSON.parse(fs.readFileSync(ROOT + "/reports/clusters.json"));
const CARD = JSON.parse(fs.readFileSync(ROOT + "/models/model_card.json"));

const FONT = "Arial";
const ACCENT = "1F4E79";
const CONTENT_W = 9638; // A4 com margens de 2 cm (11906 - 2*1134)

// ---------- formatação numérica pt-BR ----------
const n = (v, d = 3) => Number(v).toFixed(d).replace(".", ",");
const pct = (v, d = 1) => (Number(v) * 100).toFixed(d).replace(".", ",") + "%";
const int = (v) => Number(v).toLocaleString("pt-BR");

// ---------- helpers de texto ----------
function runs(text, base = {}) {
  // **negrito** e `código`
  const out = [];
  const re = /(\*\*[^*]+\*\*|`[^`]+`)/g;
  let last = 0, m;
  while ((m = re.exec(text))) {
    if (m.index > last) out.push(new TextRun({ text: text.slice(last, m.index), font: FONT, ...base }));
    const t = m[0];
    if (t.startsWith("**")) out.push(new TextRun({ text: t.slice(2, -2), bold: true, font: FONT, ...base }));
    else out.push(new TextRun({ text: t.slice(1, -1), font: "Consolas", size: 19, ...base }));
    last = m.index + t.length;
  }
  if (last < text.length) out.push(new TextRun({ text: text.slice(last), font: FONT, ...base }));
  return out;
}
const P = (text, opts = {}) => new Paragraph({ children: runs(text, opts.run || {}), spacing: { after: 120, line: 300 },
  alignment: opts.align || AlignmentType.JUSTIFIED, ...(opts.para || {}) });
const H1 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun({ text: t, font: FONT })], pageBreakBefore: true });
const H1nb = (t) => new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun({ text: t, font: FONT })] });
const H2 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun({ text: t, font: FONT })] });
const H3 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_3, children: [new TextRun({ text: t, font: FONT })] });
const B = (text, level = 0) => new Paragraph({ numbering: { reference: "bullets", level }, children: runs(text),
  spacing: { after: 60, line: 288 } });
const NUM = (text, ref = "steps") => new Paragraph({ numbering: { reference: ref, level: 0 }, children: runs(text),
  spacing: { after: 60, line: 288 } });
const NOTE = (text) => new Paragraph({ children: runs(text, { italics: true, color: "52514E", size: 20 }),
  spacing: { after: 160 } });
const CODE = (lines) => lines.map((l, i) => new Paragraph({
  children: [new TextRun({ text: l || " ", font: "Consolas", size: 18 })],
  shading: { type: ShadingType.CLEAR, fill: "F3F3F1", color: "auto" },
  spacing: { after: 0, before: i === 0 ? 80 : 0, line: 240 },
  ...(i === lines.length - 1 ? { spacing: { after: 160, line: 240 } } : {}),
}));

// ---------- tabelas ----------
let tabN = 0;
function TABLE(title, reading, header, rows, widths, opts = {}) {
  tabN += 1;
  const total = widths.reduce((a, b) => a + b, 0);
  const border = { style: BorderStyle.SINGLE, size: 4, color: "BFBFBF" };
  const borders = { top: border, bottom: border, left: border, right: border };
  const mk = (cells, isHead) => new TableRow({
    tableHeader: isHead,
    cantSplit: true,
    children: cells.map((c, i) => new TableCell({
      borders, width: { size: widths[i], type: WidthType.DXA },
      shading: isHead ? { type: ShadingType.CLEAR, fill: "DCE6F1", color: "auto" }
        : (opts.highlight && opts.highlight(cells) ? { type: ShadingType.CLEAR, fill: "FFF4D6", color: "auto" } : undefined),
      margins: { top: 60, bottom: 60, left: 100, right: 100 },
      children: String(c).split("\n").map((line) => new Paragraph({
        children: runs(line, { size: 18, bold: isHead }),
        alignment: (opts.align && opts.align[i]) || AlignmentType.LEFT,
      })),
    })),
  });
  return [
    new Paragraph({ children: [new TextRun({ text: `Tabela ${tabN} — ${title}`, bold: true, font: FONT, size: 20 })],
      spacing: { before: 160, after: 80 }, keepNext: true }),
    new Table({ width: { size: total, type: WidthType.DXA }, columnWidths: widths,
      rows: [mk(header, true), ...rows.map((r) => mk(r, false))] }),
    new Paragraph({ children: runs("Leitura: " + reading, { italics: true, size: 18, color: "52514E" }),
      spacing: { before: 60, after: 200 } }),
  ];
}

// ---------- figuras ----------
let figN = 0;
function FIGURE(file, title, reading, widthIn = 6.3) {
  figN += 1;
  const buf = fs.readFileSync(FIG + file);
  const w = buf.readUInt32BE(16), h = buf.readUInt32BE(20);
  const wpx = Math.round(widthIn * 96);
  const hpx = Math.round(wpx * h / w);
  return [
    new Paragraph({ alignment: AlignmentType.CENTER, keepNext: true, spacing: { before: 120, after: 60 },
      children: [new ImageRun({ type: "png", data: buf, transformation: { width: wpx, height: hpx },
        altText: { title, description: title, name: file } })] }),
    new Paragraph({ alignment: AlignmentType.CENTER, keepNext: true, spacing: { after: 40 },
      children: [new TextRun({ text: `Figura ${figN} — ${title}`, bold: true, font: FONT, size: 19 })] }),
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 220 },
      children: runs(reading, { italics: true, size: 18, color: "52514E" }) }),
  ];
}

// ======================================================================================
const r = CV.resultados;
const mt = M.metricas;
const pc = mt.por_classe;
const al = M.alerta_negativo;
const cm = M.matriz_confusao.valores;
const REPO = "github.com/EduBremm/reviewlens";

const children = [];

// ---------------- CAPA ----------------
children.push(
  new Paragraph({ spacing: { before: 1400, after: 200 }, alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: "CENTRO UNIVERSITÁRIO FAG — ENGENHARIA DE SOFTWARE", font: FONT, size: 22, color: "52514E" })] }),
  new Paragraph({ spacing: { after: 120 }, alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: "Projeto de Mineração de Dados (CRISP-DM + MLOps)", font: FONT, size: 24, color: "52514E" })] }),
  new Paragraph({ spacing: { before: 600, after: 200 }, alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: "ReviewLens", font: FONT, size: 64, bold: true, color: ACCENT })] }),
  new Paragraph({ spacing: { after: 120 }, alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: "App Review Intelligence", font: FONT, size: 32, color: ACCENT })] }),
  new Paragraph({ spacing: { after: 900 }, alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: "Tema 9 — Mineração de Sentimentos em Reviews de Aplicativos", font: FONT, size: 26 })] }),
);
children.push(...TABLE("Identificação do projeto", "dados de capa exigidos pelo modelo, com links do repositório e do vídeo de demonstração.",
  ["Campo", "Preenchimento"],
  [
    ["Tema (nº e título)", "9 — Mineração de Sentimentos em Reviews de Aplicativos"],
    ["Nome do produto", "ReviewLens — App Review Intelligence"],
    ["Equipe", "Eduardo Luis Bremm (trabalho individual)"],
    ["Disciplina / professor", "Engenharia de Software — Projetos Práticos (Mineração de Dados) / Prof. Franco"],
    ["Repositório", REPO],
    ["Dataset (fonte e licença)", "Reviews da Google Play de 15 apps de produtividade, coletadas com google-play-scraper (Valkov, 2020); espelho público no GitHub fixado no commit 0a25c92; sem licença explícita no espelho — uso acadêmico, sem dados pessoais"],
    ["Demonstração (vídeo/URL)", "https://drive.google.com/file/d/1iOHtOwXXQ-n4MrrVfy-lTfX4UWOH_NoX/view?usp=sharing"],
    ["Data de entrega", "24/11/2026"],
  ], [2800, 6838]));

// ---------------- RESUMO ----------------
children.push(new Paragraph({ children: [new PageBreak()] }));
children.push(H2("Resumo"));
children.push(P(`Lojas de aplicativos recebem milhares de avaliações em texto livre, e as equipes de produto não conseguem ler tudo para descobrir o que incomoda os usuários. Este projeto aplica a metodologia CRISP-DM a ${int(Q.linhas)} reviews da Google Play de 15 aplicativos de produtividade (2015–2020) para classificar cada review como POSITIVO, NEUTRO ou NEGATIVO e agrupar as reclamações por tema. Após limpeza e deduplicação restaram ${int(PREP.saida)} reviews, divididas em 80% treino e 20% teste estratificados (random_state=42). Comparamos sete abordagens com validação cruzada estratificada de 5 dobras e busca em grade: baseline de classe majoritária, o léxico pré-treinado VADER, Naive Bayes, SVM linear, Regressão Logística, vetores de palavra pré-treinados do spaCy e um modelo híbrido TF-IDF + VADER. A Regressão Logística com TF-IDF de unigramas e bigramas venceu (F1-macro de ${n(r.logreg.f1_macro)} ± ${n(r.logreg.f1_macro_std)} na validação cruzada) e, no conjunto de teste usado uma única vez, obteve F1-macro de ${n(mt.f1_macro)}, ROC-AUC de ${n(mt.roc_auc_ovr)} e acurácia de ${n(mt.accuracy)}, ${n(M.ganho_f1_vs_vader_pp, 1)} pontos acima do VADER. Com o limiar de alerta escolhido no treino, o sistema captura ${pct(al.recall_neg)} dos reviews negativos. Como segunda análise, o K-Means separou as reclamações em ${CL.k_escolhido} grupos (preço/assinatura, sincronização, notificações, anúncios, entre outros). O modelo foi empacotado em um pipeline reprodutível (DVC + MLflow), servido por uma API FastAPI com persistência em SQLite e exibido em um dashboard Streamlit com sentimento, evolução temporal, principais reclamações, termos frequentes e versões com maior insatisfação.`));
children.push(P("**Palavras-chave:** análise de sentimentos; reviews de aplicativos; CRISP-DM; classificação de texto; MLOps."));

// ---------------- SUMÁRIO ----------------
children.push(new Paragraph({ children: [new PageBreak()] }));
children.push(new Paragraph({ children: [new TextRun({ text: "Sumário", font: FONT, size: 28, bold: true, color: ACCENT })], spacing: { after: 200 } }));
children.push(new TableOfContents("Sumário", { hyperlink: true, headingStyleRange: "1-2" }));
children.push(NOTE("No Word, clique com o botão direito no sumário e escolha “Atualizar campo” para preencher os números de página."));

// ---------------- 1 ----------------
children.push(H1("1. Entendimento do negócio"));
children.push(NOTE("Critério 1 — Entendimento do problema e CRISP-DM (10 pts)."));
children.push(H2("1.1 Contexto e problema"));
children.push(P("Aplicativos populares recebem centenas de avaliações por semana na Google Play e na App Store. Cada avaliação traz uma nota de 1 a 5 estrelas e um texto livre, mas a nota sozinha não explica o problema: duas avaliações de 1 estrela podem falar de cobrança indevida e de falha de sincronização, que vão para equipes diferentes. Ler tudo manualmente é lento, e a leitura por amostragem tende a destacar o que é mais recente ou mais barulhento."));
children.push(P("O objetivo do projeto é transformar esse texto não estruturado em indicadores de satisfação: o sentimento de cada review, como ele evolui no tempo e por versão do app, e quais são os temas recorrentes das reclamações. O classificador trabalha só com o texto, o que permite aplicá-lo também a fontes sem nota (e-mails de suporte, redes sociais, respostas de pesquisa) e detectar reviews cuja nota contradiz o texto."));
children.push(H2("1.2 Objetivos"));
children.push(...TABLE("Objetivos e critérios de sucesso (definidos antes da avaliação no teste)",
  "as metas de métrica foram fixadas antes de usar o conjunto de teste; a seção 5.1 confronta cada meta com o resultado obtido.",
  ["Tipo", "Definição"],
  [
    ["Objetivo de negócio", "Reduzir o tempo para a equipe de produto identificar insatisfação e seus motivos, priorizando correções por tema e por versão"],
    ["Objetivo de mineração", "Classificação multiclasse do texto: NEGATIVO (1–2★), NEUTRO (3★), POSITIVO (4–5★); segunda análise: agrupamento (K-Means) dos reviews negativos"],
    ["Usuário da solução", "Gerente de produto, analista de suporte/CX e desenvolvedores do app"],
    ["Decisão apoiada", "O que corrigir primeiro (tema de reclamação), se uma versão deve ser investigada ou revertida, e quais reviews responder com prioridade (alerta de negativo)"],
    ["Critério de sucesso (métrica)", "F1-macro ≥ 0,60 e ROC-AUC (um-contra-todos, macro) ≥ 0,80 no teste; ganho ≥ 10 p.p. de F1-macro sobre o léxico VADER"],
    ["Critério de sucesso (negócio)", "Fila de alerta com recall da classe NEGATIVO ≥ 0,90 e precisão ≥ 0,60 (perder poucas reclamações sem inundar o suporte)"],
  ], [2800, 6838]));
children.push(P("O F1-macro foi escolhido como métrica principal porque dá o mesmo peso às três classes. Com 16% de neutros, a acurácia premiaria um modelo que ignorasse essa classe; o F1-macro não."));
children.push(H2("1.3 Premissas, restrições e riscos"));
children.push(B("**Rótulo fraco.** O sentimento é derivado das estrelas (1–2 negativo, 3 neutro, 4–5 positivo). Há reviews com nota alta e texto de reclamação, e vice-versa; isso coloca um teto no desempenho atingível e afeta principalmente a classe NEUTRO."));
children.push(B("**Amostra não representativa da distribuição real.** O coletor buscou um número fixo de reviews por app e por nota. A proporção de notas no dataset (Figura 1) não é a proporção real da loja e, como as notas baixas são mais raras, os reviews negativos vêm de um período mais longo. Por isso a queda do percentual de negativos ao longo de 2019–2020 é efeito da coleta, não uma tendência real (seção 2.3)."));
children.push(B("**Idioma e domínio.** Os textos estão em inglês e cobrem só apps de produtividade (tarefas, hábitos, calendário). Os resultados não se transferem sem retreino para reviews em português ou para outras categorias."));
children.push(B("**Privacidade.** O arquivo bruto contém nome e foto do usuário; essas colunas são descartadas na ingestão e nunca chegam ao banco nem à API."));
children.push(B("**Vazamento.** A resposta do desenvolvedor (`replyContent`, `repliedAt`) é posterior ao review e foi excluída dos atributos."));
children.push(H2("1.4 Plano CRISP-DM"));
children.push(P("O projeto percorreu as seis fases do CRISP-DM em cinco ciclos, voltando a fases anteriores sempre que a avaliação revelou um problema. A Tabela 3 registra cada retorno e o que mudou."));
children.push(...TABLE("Iterações do ciclo CRISP-DM", "cada linha é um retorno a uma fase anterior motivado por evidência; os números vêm dos relatórios gerados pelo pipeline.",
  ["#", "Onde se percebeu", "Problema", "Retorno e ajuste"],
  [
    ["1", "Entendimento dos dados", `${int(Q.texto_duplicado)} textos duplicados; ${Q.textos_duplicados_com_notas_diferentes} textos idênticos com notas diferentes`, "Preparação: remove cópias exatas e descarta os textos com rótulo contraditório (246 linhas)"],
    ["2", "Modelagem (ablação)", `Stopwords padrão removem “not”, “no”, “never”: F1-macro ${n(CV.ablacao_stopwords["stopwords padrão sklearn (remove not/no/never)"].f1_macro)}`, `Preparação: testada lista que preserva negações (${n(CV.ablacao_stopwords["stopwords customizada (mantém negações) — iteração 1"].f1_macro)}) e, por fim, nenhuma remoção (${n(CV.ablacao_stopwords["sem remoção de stopwords — adotada (iteração 2)"].f1_macro)}) — adotada`],
    ["3", "Modelagem", "Melhor C da regressão logística na borda da grade (0,5)", "Grade ampliada para C ∈ {0,1; 0,25; 0,5; 1; 2; 5}; o ótimo se manteve em 0,5"],
    ["4", "Avaliação (agrupamento)", "Silhueta cresce sem pico com k; um grupo concentrava 48% das reclamações", "Preparação: reviews negativos com ≥ 5 palavras, SVD de 50 componentes, k limitado a 6–12"],
    ["5", "Implantação", "Dashboard exibia nomes de app antigos: etapa load_db não dependia da tabela limpa", "Pipeline: dependências do DVC corrigidas; banco marcado como persistente para não apagar o histórico de predições"],
  ], [500, 2000, 3400, 3738]));

// ---------------- 2 ----------------
children.push(H1("2. Entendimento dos dados"));
children.push(NOTE("Critério 2 — Aquisição, qualidade e preparação (15 pts, junto com a seção 3)."));
children.push(H2("2.1 Fonte e aquisição"));
children.push(...TABLE("Fonte e aquisição dos dados", "o download é automático e o hash SHA-256 garante que todos usam o mesmo arquivo.",
  ["Item", "Descrição"],
  [
    ["Fonte", "Reviews públicos da Google Play coletados com a biblioteca google-play-scraper (conjunto divulgado por Valkov, 2020); espelho em github.com/kavyasri0804/Sentiment-Analysis-on-Google-Play-Store-Data-using-NLP"],
    ["Forma de coleta", "Script `src/data/download.py` baixa a URL fixada no commit 0a25c92 e confere o SHA-256 definido em `params.yaml`; sem download manual"],
    ["Volume", `${int(Q.linhas)} linhas × ${Q.colunas} colunas; ${Q.apps} aplicativos`],
    ["Período / versão", `${Q.periodo[0]} a ${Q.periodo[1]}; SHA-256 3e7ad924…865c; ${pct(PREP.saida > 0 ? 0.7465 : 0)} dos reviews são de 2020`],
    ["Licença / uso", "Dados públicos da loja; o espelho não declara licença. Uso exclusivamente acadêmico; nome e foto do autor removidos na ingestão"],
  ], [2400, 7238]));
children.push(P("**Por que este conjunto.** O enunciado pede Google Play, App Store ou Kaggle. O ambiente de desenvolvimento não tinha acesso de rede à Google Play nem ao Kaggle, então foi escolhido um conjunto já coletado com o scraper oficial, acessível por URL estável. Ele traz exatamente os campos que o produto pede: texto, nota, data e versão do app que o usuário tinha instalada. Para trocar por uma coleta própria (por exemplo, apps brasileiros em português), basta substituir a etapa de download por uma chamada ao google-play-scraper que grave o mesmo esquema; o resto do pipeline não muda."));
children.push(H2("2.2 Dicionário de dados"));
children.push(...TABLE("Dicionário de dados do arquivo bruto", "colunas marcadas como “descartada” não seguem para a preparação.",
  ["Coluna", "Tipo", "Descrição", "Exemplo / uso"],
  [
    ["reviewId", "texto (id)", "Identificador único do review", "gp:AOqpTOEh… — chave primária"],
    ["userName", "texto", "Nome público do autor", "descartada (PII)"],
    ["userImage", "texto (URL)", "Foto do autor", "descartada (PII)"],
    ["content", "texto", "Texto do review — entrada do modelo", "“I cannot open the app anymore”"],
    ["score", "inteiro 1–5", "Nota em estrelas — origem do rótulo", "1"],
    ["thumbsUpCount", "inteiro", "Votos de “útil” de outros usuários", "0 — ordena exemplos no dashboard"],
    ["reviewCreatedVersion", "texto", "Versão do app instalada no review", "5.4.0.6 → versão 5.4"],
    ["at", "data-hora", "Data do review", "2020-10-27 21:24:41"],
    ["replyContent", "texto", "Resposta do desenvolvedor", "descartada (posterior ao review)"],
    ["repliedAt", "data-hora", "Data da resposta", "descartada (posterior ao review)"],
    ["sortOrder", "texto", "Ordenação usada na coleta", "sempre “newest” — descartada"],
    ["appId", "texto", "Pacote do aplicativo", "com.todoist → Todoist"],
    ["label (derivada)", "categórica (alvo)", "Sentimento a partir da nota", "NEGATIVO / NEUTRO / POSITIVO"],
    ["version_major, month, n_words, app_name (derivadas)", "diversas", "Atributos para o dashboard e para a análise de erros", "5.4 / 2020-10 / 7 / Todoist"],
  ], [2300, 1500, 3100, 2738]));
children.push(H2("2.3 Análise exploratória (EDA)"));
children.push(P("As figuras abaixo são geradas por `python -m src.analysis.eda` sobre a base já limpa (11.514 reviews)."));
children.push(...FIGURE("eda_alvo.png", "Distribuição das notas e da classe-alvo",
  "As notas estão quase equilibradas por causa da coleta por cota; após o mapeamento, NEUTRO é a classe minoritária (16,2%)."));
children.push(...FIGURE("eda_comprimento.png", "Comprimento do review por sentimento",
  "Reviews positivos são mais curtos (mediana de 16 palavras); negativos e neutros explicam mais (24 e 25 palavras)."));
children.push(...FIGURE("eda_temporal.png", "Volume mensal e percentual de negativos (2019–2020)",
  "75% dos reviews são de 2020. O percentual de negativos cai de 84% (jan/2019) para 20% (out/2020): efeito da coleta por cota de nota, não melhora real dos apps."));
children.push(...FIGURE("eda_apps.png", "Mix de sentimento por aplicativo",
  "A maioria dos apps tem 40–47% de negativos (efeito da cota); HabitNow e LifeRPG destoam com menos de 20%, porque têm menos reviews antigos na amostra."));
children.push(...FIGURE("eda_correlacao.png", "Correlação de Spearman entre atributos numéricos",
  "Nenhum atributo numérico prevê bem a nota (|ρ| ≤ 0,21): o sinal está no texto. Reviews negativos são mais longos e recebem mais votos de “útil”.", 4.6));
children.push(...FIGURE("eda_termos.png", "Termos mais frequentes por classe (sem stopwords)",
  "Frequência bruta é parecida entre classes (“app”, “time”, “tasks”); por isso o modelo usa TF-IDF e bigramas, que ponderam termos discriminativos."));
children.push(P("**Principais achados.** (1) O alvo é moderadamente desbalanceado (40,8% / 16,2% / 43,0%), com a classe neutra como a mais difícil; (2) o comprimento e os votos de utilidade carregam pouco sinal, então o modelo precisa do conteúdo do texto; (3) reviews negativos citam versão com mais frequência (26,5% sem versão contra 8,5% nos positivos), o que viabiliza o indicador de versões com maior insatisfação; (4) a série temporal reflete a forma de coleta e deve ser lida com cuidado no dashboard."));
children.push(H2("2.4 Relatório de qualidade"));
children.push(...TABLE("Relatório de qualidade dos dados brutos (reports/quality.json)", "os problemas encontrados são tratados na preparação (seção 3) e nenhum invalida o esquema.",
  ["Verificação", "Resultado", "Ação"],
  [
    ["Esquema e tipos", "6 colunas obrigatórias presentes; score em 1..5; datas válidas", "Validação automática; pipeline para se falhar"],
    ["Valores ausentes", "content 0%; reviewCreatedVersion 17,3%; replyContent 53,4%", "Versão ausente vira “sem versão”; replyContent descartada"],
    ["Duplicados", `reviewId 0; texto ${int(Q.texto_duplicado)} (6,1%), dos quais ${Q.textos_duplicados_com_notas_diferentes} com notas diferentes`, "Cópias removidas; textos com rótulo contraditório descartados (246 linhas)"],
    ["Textos vazios/curtos", `${Q.texto_vazio_ou_curto_lt3} com menos de 3 caracteres`, "Removidos"],
    ["Outliers de tamanho", `${Q.outliers_comprimento_iqr.qtd} reviews (5,2%) acima de ${Q.outliers_comprimento_iqr.limite_superior} caracteres (regra IQR)`, "Mantidos: textos longos são legítimos; TF-IDF com sublinear_tf limita o peso"],
    ["Ruído de codificação", `${n(Q.nao_ascii_pct, 2)}% com caracteres não ASCII (emojis, aspas tipográficas)`, "Normalização Unicode NFKD no pré-processamento"],
    ["Vazamento de dados", "replyContent e repliedAt são posteriores ao review", "Excluídas dos atributos"],
    ["Dados pessoais", "userName, userImage", "Excluídas na ingestão"],
    ["Desbalanceamento do alvo", "40,8% / 16,2% / 43,0%", "Divisão estratificada, class_weight balanceado e F1-macro (seção 4)"],
  ], [2300, 3800, 3538]));

// ---------------- 3 ----------------
children.push(H1("3. Preparação dos dados"));
children.push(H2("3.1 Divisão dos dados"));
children.push(P(`Divisão aleatória 80% treino / 20% teste, estratificada pelo rótulo, com random_state=42 (\`params.yaml\`): ${int(PREP.treino)} reviews de treino e ${int(PREP.teste)} de teste, com a mesma proporção de classes nos dois conjuntos (diferença máxima de 0,02 p.p., verificada em teste automatizado). O teste é lido uma única vez, na etapa de avaliação. Como o tema não é de série temporal, a divisão principal é aleatória; uma validação temporal complementar (treina antes de agosto/2020, testa depois) aparece na seção 5.5.`));
children.push(H2("3.2 Transformações"));
children.push(P("Toda transformação que aprende algo com os dados (vocabulário e pesos do TF-IDF, padronização) fica dentro de um `Pipeline` do scikit-learn. Na validação cruzada, ela é ajustada só nas dobras de treino, sem vazamento para a dobra de validação. A API recebe o texto original e aplica exatamente o mesmo pipeline."));
children.push(...TABLE("Transformações aplicadas, em ordem", "as etapas 1–4 são determinísticas (build.py); as etapas 5–7 são aprendidas e vivem dentro do Pipeline do modelo.",
  ["Etapa", "Técnica", "Colunas", "Justificativa"],
  [
    ["1. Seleção", "Mantém 7 colunas", "todas", "Remove PII e atributos posteriores ao review"],
    ["2. Limpeza", "Remove textos < 3 caracteres, duplicados e rótulos contraditórios", "content, score", "Ruído e ambiguidade de rótulo"],
    ["3. Rótulo", "1–2★ → NEGATIVO, 3★ → NEUTRO, 4–5★ → POSITIVO", "score", "Definição do alvo (seção 1.2)"],
    ["4. Atributos de apoio", "versão major (5.4.0.6 → 5.4), mês, nº de palavras, nome do app", "derivadas", "Indicadores do dashboard e análise de erros"],
    ["5. Normalização de texto", "minúsculas, NFKD→ASCII, remove URL/e-mail/números, junta contrações (can't → cant)", "content", "Reduz variações do mesmo token"],
    ["6. Tokenização e vetorização", "TF-IDF, unigramas+bigramas, min_df=2, max_df=0,95, sublinear_tf", "content", "Bigramas capturam “not working”, “but the”"],
    ["7. Stopwords", "Nenhuma removida (ablação, Tabela 10)", "content", "Negações e “but” carregam sentimento"],
    ["Desbalanceamento", "class_weight=\"balanced\" (LR, SVM, spaCy+LR) e estratificação", "label", "Melhora o recall da classe NEUTRO sem gerar dados sintéticos"],
    ["Representações pré-treinadas", "Léxico VADER (4 escores); vetores spaCy en_core_web_md (média de 300 dim.)", "content", "Componentes “pré-treinados de NLP” pedidos no tema"],
  ], [1900, 3000, 1300, 3438]));
children.push(H2("3.3 Dataset final"));
children.push(P(`${int(PREP.entrada)} linhas de entrada → ${int(PREP.apos_remover_curtos)} após remover textos curtos → ${int(PREP.apos_deduplicar)} após deduplicar e descartar rótulos contraditórios → ${int(PREP.saida)} após a limpeza de texto. Treino: ${int(PREP.treino)} reviews; teste: ${int(PREP.teste)}. Com unigramas e bigramas e min_df=2, o vocabulário do modelo final tem 29.698 termos. Arquivos em \`data/processed/\` (Parquet), versionados pelo DVC.`));

// ---------------- 4 ----------------
children.push(H1("4. Modelagem"));
children.push(NOTE("Critério 3 — Mineração e modelagem (20 pts)."));
children.push(H2("4.1 Baselines"));
children.push(P("Dois baselines definem o piso: (a) **DummyClassifier**, que sempre prevê a classe majoritária (POSITIVO); e (b) **VADER**, um léxico de sentimento pré-treinado aplicado sem nenhum treino (zero-shot), com os limiares padrão do autor (compound ≤ −0,05 negativo, ≥ 0,05 positivo). Todo modelo treinado precisa superar os dois."));
children.push(H2("4.2 Algoritmos comparados"));
children.push(...TABLE("Modelos candidatos e hiperparâmetros buscados", "cobre os quatro algoritmos sugeridos para o tema (Naive Bayes, SVM, Regressão Logística e modelos pré-treinados de NLP).",
  ["Modelo", "Por que foi escolhido", "Hiperparâmetros buscados"],
  [
    ["Complement Naive Bayes + TF-IDF", "Clássico para texto; rápido; a variante Complement lida melhor com classes desbalanceadas", "alpha ∈ {0,1; 0,3; 1}; n-gramas (1,1) ou (1,2)"],
    ["SVM linear + TF-IDF (calibrado)", "Forte em texto de alta dimensão; calibração sigmoide para gerar probabilidades", "C ∈ {0,05; 0,1; 0,5; 1}; n-gramas"],
    ["Regressão Logística + TF-IDF", "Linear, interpretável por coeficiente, probabilidades bem comportadas", "C ∈ {0,1; 0,25; 0,5; 1; 2; 5}; n-gramas"],
    ["VADER (pré-treinado, zero-shot)", "Léxico de sentimento construído para textos de redes sociais", "— (limiares padrão)"],
    ["Vetores spaCy + Regressão Logística", "Embeddings de palavra pré-treinados (300 dim.) capturam sinônimos", "C ∈ {0,01; 0,1; 1}"],
    ["Híbrido TF-IDF + VADER + LR", "Junta vocabulário do domínio com o conhecimento prévio do léxico", "C ∈ {0,25; 0,5; 1; 2}"],
  ], [2700, 4100, 2838]));
children.push(P("**Sobre transformers (BERT/RoBERTa).** Modelos pré-treinados baseados em transformers não puderam ser baixados no ambiente (sem acesso ao Hugging Face). O tema foi atendido com dois modelos pré-treinados distribuídos como pacote Python (VADER e vetores do spaCy), e a troca por um transformer está prevista em trabalhos futuros — a interface `Pipeline` permite plugá-lo sem mudar API nem dashboard."));
children.push(H2("4.3 Protocolo de validação"));
children.push(P("`StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` aplicado só ao conjunto de treino; `GridSearchCV` com busca exaustiva na grade da Tabela 8 (37 combinações no total), otimizando F1-macro e registrando também acurácia, acurácia balanceada, ROC-AUC (um-contra-todos) e recall de cada classe. Cada candidato gera um run no MLflow com parâmetros, métricas e hash do commit."));
children.push(H2("4.4 Resultados na validação cruzada"));
const order = ["dummy", "vader", "naive_bayes", "svm", "spacy_vectors_lr", "hybrid_tfidf_vader", "logreg"];
const names = { dummy: "Baseline (classe majoritária)", vader: "VADER (zero-shot)", naive_bayes: "Naive Bayes", svm: "SVM linear",
  spacy_vectors_lr: "Vetores spaCy + LR", hybrid_tfidf_vader: "Híbrido TF-IDF + VADER", logreg: "Regressão Logística ★" };
children.push(...TABLE("Comparação dos modelos — validação cruzada de 5 dobras no treino (média ± desvio, seed 42)",
  "a Regressão Logística tem o maior F1-macro; o híbrido empata dentro do desvio, mas é mais complexo; o SVM tem a maior acurácia porque quase ignora a classe NEUTRO (recall 0,09).",
  ["Modelo", "F1-macro", "ROC-AUC", "Acurácia", "Recall NEG", "Recall NEU", "Recall POS", "Fit (s)"],
  order.map((k) => [names[k], `${n(r[k].f1_macro)} ± ${n(r[k].f1_macro_std)}`, `${n(r[k].roc_auc_ovr)} ± ${n(r[k].roc_auc_ovr_std)}`,
    n(r[k].accuracy), n(r[k].recall_neg), n(r[k].recall_neu), n(r[k].recall_pos), n(r[k].tempo_fit_medio_s, 2)]),
  [2300, 1250, 1250, 950, 950, 950, 950, 1038],
  { highlight: (c) => String(c[0]).includes("★"), align: [AlignmentType.LEFT, ...Array(7).fill(AlignmentType.RIGHT)] }));
children.push(P("Três observações: (1) os modelos baseados em TF-IDF superam os pré-treinados genéricos, porque o vocabulário do domínio (“sync”, “premium”, “widget”) pesa mais que o sentimento geral das palavras; (2) o VADER erra muito a classe NEUTRO (recall 0,15) — o léxico classifica como positivo um texto morno com palavras como “good”; (3) a calibração sigmoide do SVM empurra a decisão para as classes grandes, o que explica a boa acurácia com F1-macro menor."));
const ab = CV.ablacao_stopwords;
children.push(...TABLE("Ablação: efeito da remoção de stopwords (Regressão Logística, CV 5 dobras)",
  "remover negações custa 4 p.p. de F1-macro; não remover nada é o melhor, e foi a configuração adotada.",
  ["Configuração", "F1-macro", "Recall NEG"],
  Object.entries(ab).map(([k, v]) => [k, `${n(v.f1_macro)} ± ${n(v.f1_macro_std)}`, n(v.recall_neg)]),
  [5638, 2200, 1800], { align: [AlignmentType.LEFT, AlignmentType.RIGHT, AlignmentType.RIGHT] }));
children.push(H2("4.5 Modelo escolhido"));
children.push(P(`**Regressão Logística + TF-IDF (unigramas e bigramas), C = 0,5, class_weight balanceado.** Foi escolhida pelo maior F1-macro na validação cruzada (${n(r.logreg.f1_macro)} ± ${n(r.logreg.f1_macro_std)}), com o melhor recall da classe NEUTRO entre os modelos fortes (${n(r.logreg.recall_neu)}). O híbrido com VADER ficou a 0,003 de distância, dentro do desvio; entre dois modelos equivalentes, preferimos o mais simples, mais rápido de servir e diretamente interpretável pelos coeficientes. O modelo foi salvo em \`models/model.joblib\`, registrado no MLflow Model Registry como “reviewlens-sentimento” e descrito em \`models/model_card.json\` (versão ${CARD.versao}).`));
children.push(H3("Segunda análise: agrupamento das reclamações"));
children.push(P(`Para responder “do que os usuários reclamam”, os ${int(CL.n_reviews_negativos)} reviews negativos com 5 ou mais palavras passaram por TF-IDF (com stopwords e palavras genéricas de review removidas), redução LSA (TruncatedSVD, 50 componentes) e K-Means. O número de grupos foi escolhido pela silhueta (cosseno), conferida com o índice de Davies-Bouldin.`));
children.push(...FIGURE("cluster_escolha_k.png", "Escolha do número de grupos (k)",
  "A silhueta cresce de forma contínua com k e o Davies-Bouldin cai — não há um “cotovelo” natural, o que é comum em texto curto. Limitamos k a 12 por legibilidade no dashboard e usamos o melhor k nesse intervalo."));
children.push(...TABLE("Grupos de reclamação encontrados (k = 12, seed 42)", "os nomes foram dados após ler os termos característicos e exemplos de cada grupo; o maior grupo reúne falhas gerais sem tema dominante.",
  ["Grupo", "Reviews", "%", "Termos característicos"],
  [...CL.clusters].sort((a, b) => b.n - a.n).map((c) => [c.nome, int(c.n), n(c.pct, 1).replace(",0", ",0"), c.top_termos.slice(0, 6).join(", ")]),
  [3300, 900, 700, 4738], { align: [AlignmentType.LEFT, AlignmentType.RIGHT, AlignmentType.RIGHT, AlignmentType.LEFT] }));
children.push(P("A silhueta final (0,13) é baixa em termos absolutos, como esperado para texto curto, mas os grupos são interpretáveis e úteis para o negócio: preço e assinatura somam três grupos (Recursos bloqueados na versão gratuita, Assinatura premium, Versão Pro do Forest — 20% das reclamações), seguidos de integração com calendário, notificações e anúncios. A API usa esse mesmo modelo para atribuir um grupo a cada novo review negativo."));

// ---------------- 5 ----------------
children.push(H1("5. Avaliação e interpretação"));
children.push(NOTE("Critério 4 — Avaliação e interpretação (10 pts)."));
children.push(H2("5.1 Resultado no teste"));
children.push(P(`O conjunto de teste (${int(M.n_teste)} reviews, 20%, estratificado, seed 42) foi usado uma única vez, com o modelo final já escolhido. O VADER foi medido no mesmo teste apenas como referência — nenhuma decisão foi tomada com base nele.`));
children.push(...TABLE("Metas da seção 1.2 versus resultado no teste", "todas as cinco metas foram atingidas; a de precisão do alerta com pouca folga (0,617 para meta de 0,60).",
  ["Métrica", "Meta", "Obtido (teste)", "Atingiu?"],
  [
    ["F1-macro", "≥ 0,60", n(mt.f1_macro), "Sim"],
    ["ROC-AUC (OvR, macro)", "≥ 0,80", n(mt.roc_auc_ovr), "Sim"],
    ["Ganho de F1-macro sobre o VADER", "≥ 10 p.p.", `${n(M.ganho_f1_vs_vader_pp, 1)} p.p. (VADER: ${n(M.vader_mesmo_teste.f1_macro)})`, "Sim"],
    ["Recall NEGATIVO no alerta (limiar 0,25)", "≥ 0,90", n(al.recall_neg), "Sim"],
    ["Precisão NEGATIVO no alerta", "≥ 0,60", n(al.precisao_neg), "Sim"],
  ], [3800, 1500, 2800, 1538], { align: [AlignmentType.LEFT, AlignmentType.CENTER, AlignmentType.CENTER, AlignmentType.CENTER] }));
children.push(...TABLE("Métricas completas no teste", "o desempenho é bom nas classes polares (F1 0,74 e 0,78) e fraco na NEUTRO (0,35), que puxa o F1-macro para baixo.",
  ["Classe / agregado", "Precisão", "Recall", "F1", "PR-AUC", "Suporte"],
  [
    ...["NEGATIVO", "NEUTRO", "POSITIVO"].map((c) => [c, n(pc[c].precision), n(pc[c].recall), n(pc[c]["f1-score"]), n(mt.pr_auc_por_classe[c]), int(pc[c].support)]),
    ["Macro", "—", n(mt.balanced_accuracy), n(mt.f1_macro), n(mt.pr_auc_macro), int(M.n_teste)],
    ["Acurácia / log-loss / Brier médio", n(mt.accuracy), "—", "—", `${n(mt.log_loss)} / ${n(mt.brier_medio)}`, ""],
  ], [3000, 1300, 1300, 1300, 1500, 1238], { align: [AlignmentType.LEFT, ...Array(5).fill(AlignmentType.RIGHT)] }));
children.push(P(`O F1-macro no teste (${n(mt.f1_macro)}) ficou dentro de um desvio da validação cruzada (${n(r.logreg.f1_macro)} ± ${n(r.logreg.f1_macro_std)}), sinal de que não houve sobreajuste na seleção do modelo.`));
children.push(H2("5.2 Figuras de avaliação"));
children.push(...FIGURE("eval_confusao.png", "Matriz de confusão no teste (limiar padrão: maior probabilidade)",
  `Os erros se concentram em torno de NEUTRO; a inversão grosseira de polaridade (negativo ↔ positivo) acontece em ${pct(M.erros.polaridade_invertida_pct)} do teste (${M.erros.polaridade_invertida} reviews).`));
children.push(...FIGURE("eval_roc_pr.png", "Curvas ROC e Precisão-Recall por classe",
  "NEGATIVO e POSITIVO têm AUC acima de 0,86; NEUTRO fica em torno de 0,70 e com PR-AUC de 0,28 — a classe intermediária é a menos separável."));
children.push(...FIGURE("eval_calibracao.png", "Curva de calibração no teste",
  "NEGATIVO e POSITIVO ficam acima da diagonal a partir de 0,4 (o modelo é conservador: com 0,6 previsto, ~78% acertam); NEUTRO fica abaixo (probabilidade superestimada), efeito do class_weight balanceado e da regularização. O dashboard mostra a probabilidade como “confiança relativa”, não como frequência exata.", 4.4));
children.push(...FIGURE("eval_importancia.png", "Importância global: termos com maior coeficiente por classe",
  "Negações (“not”, “no”, “cant”, “doesnt”) e “useless”, “uninstalled”, “waste” empurram para NEGATIVO; “but”, “ok”, “average” para NEUTRO; “love”, “great”, “best”, “easy” para POSITIVO."));
children.push(H2("5.3 Escolha do limiar e impacto no negócio"));
children.push(P(`Para a fila de alerta do suporte, perder uma reclamação custa mais do que revisar um falso alarme. Por isso o alerta usa um limiar próprio sobre P(NEGATIVO), escolhido no treino com predições out-of-fold (5 dobras): o maior limiar que mantém recall ≥ 0,90. O resultado foi **0,25** (no treino, recall 0,947 e precisão 0,612). No teste, esse limiar capturou ${pct(al.recall_neg)} dos reviews negativos (perdeu ${al.negativos_perdidos} de ${al.negativos_total}) com precisão de ${pct(al.precisao_neg)}.`));
children.push(...TABLE("Curva de limiar do alerta de negativo (predições out-of-fold no treino)", "subir o limiar reduz a fila, mas perde reclamações; 0,25 é o último ponto com recall acima de 0,90.",
  ["Limiar P(NEG)", "Recall NEG", "Precisão NEG", "% de reviews na fila"],
  CV.curva_limiar_neg_oof.filter((c) => c.limiar <= 0.5).map((c) => [n(c.limiar, 2), n(c.recall_neg), n(c.precisao_neg), pct(c.alertas_pct)]),
  [2400, 2400, 2400, 2438], { highlight: (c) => c[0] === "0,25", align: Array(4).fill(AlignmentType.CENTER) }));
children.push(P(`**Impacto.** A fila de alerta contém ${pct(al.alertas_pct)} dos reviews — alto, porque na amostra 41% dos reviews são negativos. Numa loja real, com proporção menor de negativos, a fila encolhe proporcionalmente. Mesmo assim, a equipe deixa de ler 38% dos reviews (os que o modelo dá como claramente não negativos) perdendo apenas 6% das reclamações, e cada reclamação chega já marcada com o grupo temático.`));
children.push(H2("5.4 Interpretação"));
children.push(P("O modelo aprendeu padrões coerentes com o domínio. Para NEGATIVO, os termos mais fortes são negações e verbos de abandono (“uninstalled”, “lost”), além de temas típicos de reclamação (“account”, “ads”, “now” — como em “now it doesn't work”). A classe NEUTRO é sinalizada por contraste: “but” é o termo de maior peso, seguido de “ok”, “average”, “app but” e “but not” — o review neutro típico elogia e critica na mesma frase (“good app but the alarm doesn't work”). Isso explica por que remover stopwords piorava o modelo: “but” e “not” estão na lista padrão. Para POSITIVO, adjetivos de elogio e “helps”, “life” (“changed my life”)."));
children.push(P("A API devolve, para cada review, os termos do próprio texto com maior contribuição (TF-IDF × coeficiente) para a classe prevista. No exemplo “After the last update the app crashes and my tasks do not sync anymore”, a resposta é NEGATIVO (0,75) com termos “not”, “crashes”, “sync”, “not sync”, “after” e grupo “Problemas após atualização”."));
children.push(H2("5.5 Análise de erros e viés"));
children.push(...FIGURE("eval_erros.png", "Acurácia no teste por tamanho do review e por nota original",
  "Reviews curtos (até 15 palavras) são mais fáceis (75%); acima de 100 palavras cai para 48%. Notas extremas (1★ e 5★) acertam 80–89%; 3★ fica em 41% e 4★ em 59%."));
children.push(B("**Classe neutra.** Dos 373 neutros do teste, 136 foram previstos como negativos e 85 como positivos. Muitos são, de fato, ambíguos: o texto é uma reclamação, mas o usuário deu 3 estrelas. É um limite do rótulo derivado da nota, não só do modelo."));
children.push(B("**Notas 2 e 4.** Concentram boa parte dos erros fora do neutro: reviews de 4★ com pedido de melhoria (“would be 5 stars if…”) e de 2★ com elogio parcial."));
children.push(B("**Textos longos.** Reviews com mais de 100 palavras misturam elogios e críticas; o TF-IDF em média perde a estrutura. São só 1,3% do teste."));
children.push(B(`**Por aplicativo.** O F1-macro varia de 0,53 (Forest) a 0,69 (Business Calendar) e 0,83 (Artful Agenda, só 32 reviews). O Forest tem vocabulário próprio (“trees”, “coins”) que confunde o modelo; os apps com mais dados ficam acima da média. A tabela completa está em reports/metrics.json.`));
children.push(B(`**Robustez temporal.** Treinando só com reviews anteriores a agosto/2020 (${int(M.robustez_temporal.n_treino)}) e testando nos posteriores (${int(M.robustez_temporal.n_teste)}), o F1-macro foi ${n(M.robustez_temporal.f1_macro)} — queda de apenas 1,4 p.p. em relação ao teste aleatório. O modelo generaliza para reviews futuros, embora a distribuição de classes mude entre os períodos.`));
children.push(B("**Viés.** O dataset não tem atributos demográficos dos autores (nome e foto foram removidos por privacidade), então não é possível medir viés por grupo de pessoas. O viés mensurável é por app e por tamanho do texto, apresentado acima: apps com poucos reviews e vocabulário particular são atendidos pior."));
children.push(P("Os 40 erros de maior confiança, com o texto, estão em `reports/errors_sample.csv` para inspeção manual."));

// ---------------- 6 ----------------
children.push(H1("6. Engenharia do pipeline (MLOps)"));
children.push(NOTE("Critério 5 — Engenharia do pipeline (15 pts)."));
children.push(H2("6.1 Etapas do pipeline"));
children.push(...TABLE("Etapas do pipeline (dvc.yaml)", "cada etapa é um módulo Python executável; o DVC liga as saídas de uma às entradas da próxima.",
  ["Etapa", "Comando", "Entrada", "Saída (artefato)"],
  [
    ["Ingestão", "python -m src.data.download", "URL fixada + SHA-256 (params.yaml)", "data/raw/reviews.csv"],
    ["Validação", "python -m src.data.validate", "data/raw/", "reports/quality.json (falha se o esquema quebrar)"],
    ["Preparação", "python -m src.features.build", "data/raw/", "data/processed/train.parquet, test.parquet, reviews_clean.parquet"],
    ["EDA", "python -m src.analysis.eda", "reviews_clean.parquet", "reports/eda.json + 6 figuras"],
    ["Treino", "python -m src.models.train", "train.parquet", "models/model.joblib, model_card.json, reports/cv_results.json, runs MLflow"],
    ["Avaliação", "python -m src.models.evaluate", "modelo + test.parquet", "reports/metrics.json, errors_sample.csv, 5 figuras"],
    ["Agrupamento", "python -m src.analysis.clustering", "reviews_clean.parquet", "models/clusters.joblib, reports/clusters.json"],
    ["Publicação", "python -m src.analysis.load_db", "modelo + reviews + grupos", "data/reviewlens.db (lido pela API e dashboard)"],
    ["Registro", "MLflow Model Registry", "run “logreg-final”", "modelo “reviewlens-sentimento” versionado"],
  ], [1400, 3000, 2300, 2938]));
children.push(H2("6.2 Orquestração e rastreamento"));
children.push(P("O `dvc.yaml` declara, para cada etapa, o comando, as dependências (código, dados, seções do `params.yaml`) e as saídas. `dvc repro` refaz só o que mudou: alterar o C da regressão logística em `params.yaml` reexecuta treino, avaliação e publicação, mas não o download nem a preparação. `dvc metrics show` lista as métricas de qualidade, preparação, CV, teste e agrupamento lado a lado, e `dvc metrics diff` compara com o commit anterior."));
children.push(P("Cada execução de treino cria um run por candidato no MLflow (backend SQLite em `mlruns/`), com parâmetros, métricas de CV (média e desvio), a seed e o hash do commit Git como tag. O run final registra o modelo no Model Registry e anexa o model card. A versão do modelo (data + hash do commit) é gravada em cada predição feita pela API."));
children.push(...FIGURE("ui_mlflow.png", "MLflow: runs do experimento reviewlens-sentimento",
  "Cada candidato aparece como um run; o run logreg-final contém o modelo registrado. Aparecem duas execuções completas do pipeline (iterações)."));
children.push(H2("6.3 Estrutura do repositório"));
children.push(...CODE([
  "reviewlens/",
  "├── data/              # raw/, processed/ (DVC) e reviewlens.db",
  "├── models/            # model.joblib, clusters.joblib, model_card.json",
  "├── reports/           # *.json de métricas, figures/, openapi.json",
  "├── src/",
  "│   ├── config.py      # caminhos e leitura do params.yaml",
  "│   ├── db.py          # esquema SQLite e gravação de predições",
  "│   ├── data/          # download.py, validate.py",
  "│   ├── features/      # text.py, transformers.py, build.py",
  "│   ├── models/        # candidates.py, train.py, evaluate.py",
  "│   ├── analysis/      # eda.py, clustering.py, insights.py, load_db.py",
  "│   └── api/           # main.py (FastAPI), schemas.py",
  "├── dashboard/app.py   # Streamlit",
  "├── tests/             # 5 arquivos, 33 testes",
  "├── params.yaml  dvc.yaml  Makefile  requirements.txt",
  "├── Dockerfile  docker-compose.yml  .github/workflows/ci.yml",
  "└── README.md",
]));
children.push(H2("6.4 Monitoramento"));
children.push(P("Toda predição feita pela API é gravada na tabela `predictions` com data, texto, probabilidades, alerta, grupo e versão do modelo, o que permite auditoria e comparação posterior com as estrelas reais. O monitoramento de drift automatizado não foi implementado; a proposta (seção 10) é um relatório semanal que compare a distribuição de classes previstas e o vocabulário novo (termos fora do vocabulário do TF-IDF) com os do treino."));

// ---------------- 7 ----------------
children.push(H1("7. Arquitetura e implementação do software"));
children.push(NOTE("Critério 6 — Arquitetura e implementação (15 pts)."));
children.push(H2("7.1 Diagrama de arquitetura"));
children.push(...FIGURE("arquitetura.png", "Arquitetura do ReviewLens — treino à esquerda, uso à direita",
  "O pipeline de treino produz os artefatos do modelo e carrega o banco; a API carrega o modelo uma vez na subida, grava cada predição e serve os indicadores que o dashboard exibe."));
children.push(H2("7.2 Pilha tecnológica"));
children.push(...TABLE("Pilha tecnológica", "versões fixadas em requirements.txt.",
  ["Camada", "Tecnologia"],
  [
    ["Linguagem", "Python 3.13"],
    ["Mineração", "pandas 3.0, scikit-learn 1.9, vaderSentiment 3.3, spaCy 3.8 (en_core_web_md)"],
    ["API", "FastAPI 0.163 + Pydantic 2 + Uvicorn"],
    ["Persistência", "SQLite (arquivo único; troca para PostgreSQL mudando a conexão em src/db.py)"],
    ["Visualização", "Streamlit 1.67 + Plotly 7; figuras do relatório em matplotlib"],
    ["MLOps", "DVC 3.67 (pipeline e métricas), MLflow 3.18 (rastreamento e Model Registry)"],
    ["Qualidade e infraestrutura", "pytest + pytest-cov, ruff, Docker Compose, GitHub Actions"],
  ], [2600, 7038]));
children.push(H2("7.3 Contrato da API"));
children.push(P("A API expõe 11 endpoints, documentados automaticamente pelo FastAPI em `/docs` (Swagger) e exportados em `reports/openapi.json` (Apêndice D)."));
children.push(...CODE([
  "POST /predict",
  "Request:",
  '{ "text": "After the last update the app crashes and my tasks do not sync anymore",',
  '  "app": "Todoist" }',
  "Response 200:",
  "{",
  '  "id": 1,',
  '  "sentimento": "NEGATIVO",',
  '  "probabilidades": { "NEGATIVO": 0.7542, "NEUTRO": 0.2021, "POSITIVO": 0.0437 },',
  '  "confianca": 0.7542,',
  '  "alerta_negativo": true,',
  '  "grupo_reclamacao": "Problemas após atualização",',
  '  "termos_influentes": ["not", "crashes", "sync", "not sync", "after"],',
  `  "model_version": "${CARD.versao}"`,
  "}",
  'Erros: texto vazio ou ausente -> 422; label inválido em /insights/terms -> 422',
]));
children.push(...TABLE("Endpoints da API", "os endpoints /insights/* aceitam ?app=<nome> para filtrar por aplicativo.",
  ["Método e rota", "Função"],
  [
    ["GET /health", "Status do serviço e do modelo carregado"],
    ["GET /model/info", "Model card: modelo, versão, hiperparâmetros, métricas de CV, limiar de alerta"],
    ["POST /predict", "Classifica um review e grava a predição"],
    ["POST /predict/batch", "Classifica até 500 reviews e devolve a distribuição"],
    ["GET /predictions?limit=", "Histórico de predições (auditoria)"],
    ["GET /apps", "Apps com volume, % negativo e nota média"],
    ["GET /insights/summary", "Totais e mix de sentimento"],
    ["GET /insights/timeline", "Sentimento por mês"],
    ["GET /insights/versions?min_reviews=", "Versões com maior % de negativos"],
    ["GET /insights/complaints", "Grupos de reclamação e tamanho"],
    ["GET /insights/terms?label=&k=", "Termos e bigramas mais frequentes por classe"],
  ], [3600, 6038]));
children.push(H2("7.4 Persistência"));
children.push(P("Banco SQLite com três tabelas (esquema em `src/db.py`):"));
children.push(B("`reviews(review_id PK, app, content, score, label_estrelas, label_modelo, p_negativo, version, version_major, at, month, thumbs_up, cluster_id)` — base histórica classificada pelo modelo, com índices por app/mês e app/versão."));
children.push(B("`complaint_clusters(cluster_id PK, nome, top_termos, n, exemplos)` — os 12 grupos de reclamação."));
children.push(B("`predictions(id PK, created_at, app, text, label, p_negativo, p_neutro, p_positivo, alerta, cluster_id, model_version)` — cada chamada à API; guardar a versão do modelo permite auditar e comparar versões."));
children.push(P("A etapa de publicação recria `reviews` e `complaint_clusters` a cada execução do pipeline, mas preserva `predictions` (saída marcada como `persist` no DVC)."));
children.push(H2("7.5 Interface / dashboard"));
children.push(P("O dashboard tem filtros por aplicativo e período, cinco indicadores no topo (volume, % negativo, % neutro, % positivo e nota média) e uma aba para cada item pedido no enunciado do tema 9. As capturas abaixo são do sistema rodando localmente."));
children.push(...FIGURE("ui_dashboard_sentimento.png", "Dashboard — aba Sentimento", "Mix de sentimento previsto pelo modelo e % de negativos por app; a concordância entre modelo e estrelas na base é de 82,9% (inclui reviews de treino).", 4.6));
children.push(...FIGURE("ui_dashboard_temporal.png", "Dashboard — aba Evolução temporal", "Participação mensal de cada sentimento e volume de reviews.", 4.6));
children.push(...FIGURE("ui_dashboard_versoes.png", "Dashboard — aba Versões", "Versões com maior insatisfação: TickTick 5.4 (90% negativos em 29 reviews), Microsoft To Do 2.17 (83%) e Habitica 1.7 (82%) lideram.", 4.6));
children.push(...FIGURE("ui_dashboard_reclamacoes.png", "Dashboard — aba Principais reclamações", "Tamanho de cada grupo do K-Means e reviews de exemplo do grupo selecionado.", 4.6));
children.push(...FIGURE("ui_dashboard_termos.png", "Dashboard — aba Termos", "Termos e bigramas mais frequentes na classe escolhida.", 4.6));
children.push(...FIGURE("ui_dashboard_classificar.png", "Dashboard — aba Classificar review", "Classificação em tempo real via API, com probabilidades, alerta, grupo, termos influentes e histórico de predições.", 4.6));

// ---------------- 8 ----------------
children.push(H1("8. Reprodutibilidade, testes e DevOps"));
children.push(NOTE("Critério 7 — Reprodutibilidade, testes e DevOps (10 pts)."));
children.push(H2("8.1 Como reproduzir"));
children.push(...CODE([
  `git clone https://${REPO}.git`,
  "cd reviewlens",
  "docker compose up --build              # API :8000, dashboard :8501, MLflow :5000",
  "docker compose run --rm app dvc repro  # dados -> treino -> avaliação -> banco",
  "",
  "# sem Docker (Python 3.13):",
  "pip install -r requirements.txt",
  "dvc repro        # ou: make pipeline-nodvc",
  "make test        # pytest + cobertura",
  "make api         # http://localhost:8000/docs",
  "make dashboard   # http://localhost:8501",
]));
children.push(P("A execução completa do pipeline leva cerca de 2 minutos e 15 segundos em uma máquina de 2 núcleos (a busca de hiperparâmetros é a etapa mais longa)."));
children.push(H2("8.2 Controles de reprodutibilidade"));
children.push(B("☑ Versões de bibliotecas fixadas com `==` em `requirements.txt` (inclusive o modelo do spaCy, por URL de release)."));
children.push(B("☑ Uma única seed (`seed: 42` em `params.yaml`) usada na divisão, na validação cruzada, nos modelos, no SVD e no K-Means."));
children.push(B("☑ Dados baixados por script de URL fixada em commit, com SHA-256 conferido; saídas versionadas pelo DVC."));
children.push(B("☑ Nenhum caminho absoluto nem passo manual: caminhos derivam da raiz do repositório (`src/config.py`)."));
children.push(B("☑ Métricas deste relatório lidas de `reports/metrics.json` e `reports/cv_results.json`; um teste automatizado recalcula o F1-macro do teste e compara com o arquivo."));
children.push(H2("8.3 Testes automatizados"));
children.push(...TABLE("Testes automatizados (33 testes, todos passando)", "os testes de modelo e API rodam depois do dvc repro no CI e usam um banco temporário, sem sujar o banco real.",
  ["Tipo", "Arquivo", "O que testa"],
  [
    ["Dados", "test_data.py", "Esquema; score fora de 1..5 é detectado; remoção de PII, curtos, duplicados e rótulos contraditórios; hash do arquivo bruto; split estratificado e sem interseção"],
    ["Transformação", "test_features.py", "Limpeza de texto (URL, números, contrações); negações preservadas; mapeamento nota → rótulo; versão major; baseline VADER"],
    ["Pipeline", "test_pipeline_parts.py", "Candidatos cobrem os algoritmos do tema; pipeline TF-IDF aceita texto bruto e devolve probabilidades válidas"],
    ["Modelo", "test_model.py", "F1-macro ≥ 0,60 e AUC ≥ 0,80 no teste; vence VADER e dummy; determinismo com a mesma seed; exemplos óbvios; relatório = modelo"],
    ["API", "test_api.py", "/health, /model/info; /predict negativo com alerta e grupo; positivo sem alerta; entradas inválidas → 422; lote e histórico; todos os /insights"],
  ], [1500, 2200, 5938]));
children.push(P("Cobertura (`pytest --cov`): **87%** dos módulos de biblioteca (features, dados, API, banco, insights). Os scripts de etapa do pipeline (treino, avaliação, EDA, agrupamento, publicação) ficam fora da medição porque são executados de ponta a ponta pelo `dvc repro` no CI, antes dos testes."));
children.push(H2("8.4 Integração contínua"));
children.push(P("O workflow `.github/workflows/ci.yml` roda a cada push na main e em cada pull request: instala as dependências fixadas, executa o lint (`ruff`), roda o pipeline completo com `dvc repro` (baixando os dados do zero), executa os testes com cobertura mínima de 80%, imprime `dvc metrics show` no log e publica a pasta `reports/` como artefato. Um segundo job constrói a imagem Docker. A regra de proteção da main deve exigir os dois jobs verdes antes do merge."));

// ---------------- 9 ----------------
children.push(H1("9. Implantação e demonstração"));
children.push(NOTE("Critério 8 — Apresentação e demonstração (5 pts)."));
children.push(H2("9.1 Ambiente de execução"));
children.push(P("Docker Compose local com três serviços a partir da mesma imagem: `api` (Uvicorn, porta 8000, com healthcheck), `dashboard` (Streamlit, porta 8501, só sobe depois que a API está saudável) e `mlflow` (porta 5000). Os volumes `data/` e `mlruns/` mantêm banco e experimentos entre reinícios. Para uma versão pública, a mesma imagem pode ser publicada no Render ou no Hugging Face Spaces."));
children.push(H2("9.2 Roteiro da demonstração (3 a 5 minutos)"));
[
  "**Problema (20 s):** equipes de produto não conseguem ler milhares de reviews; o ReviewLens diz o sentimento e o motivo.",
  "**Subida (20 s):** `docker compose up` e `curl localhost:8000/health`.",
  "**Caso negativo ponta a ponta (60 s):** na aba Classificar, colar “After the last update the app crashes and my tasks do not sync anymore” → NEGATIVO 75%, alerta, grupo “Problemas após atualização”, termos “not”, “crashes”, “sync”.",
  "**Contraste (30 s):** “Love this planner, it keeps my week organized” → POSITIVO 78%, sem alerta; e um neutro (“The widget is ok but sync is slow sometimes”).",
  "**Dashboard (60 s):** filtrar TickTick; mostrar versão 5.4 com 90% de negativos; abrir o grupo “Assinatura premium e suporte” com exemplos.",
  "**MLflow (30 s):** comparar os 7 runs e mostrar o modelo registrado.",
  "**CI (20 s):** pipeline verde no GitHub Actions com as métricas no log.",
].forEach((t) => children.push(NUM(t)));
children.push(H2("9.3 Link"));
children.push(P(`Repositório: ${REPO}. Vídeo de demonstração (2min53, Google Drive): https://drive.google.com/file/d/1iOHtOwXXQ-n4MrrVfy-lTfX4UWOH_NoX/view?usp=sharing. O vídeo percorre o roteiro da seção 9.2 com o sistema em execução e legendas explicativas.`));

// ---------------- 10 ----------------
children.push(H1("10. Conclusões, limitações e trabalhos futuros"));
children.push(H2("Conclusão"));
children.push(P(`As cinco metas da seção 1.2 foram atingidas. A Regressão Logística com TF-IDF obteve F1-macro de ${n(mt.f1_macro)} e ROC-AUC de ${n(mt.roc_auc_ovr)} no teste, ${n(M.ganho_f1_vs_vader_pp, 1)} pontos acima do léxico pré-treinado VADER, e o alerta de negativos captura ${pct(al.recall_neg)} das reclamações com precisão de ${pct(al.precisao_neg)}. O agrupamento mostrou que preço e assinatura são o tema de reclamação mais frequente depois das falhas gerais, informação diretamente acionável. O resultado está integrado em um software completo — pipeline reprodutível, API com persistência e auditoria, dashboard com os cinco indicadores pedidos — e não apenas em um notebook.`));
children.push(H2("Limitações"));
children.push(B("O rótulo vem das estrelas: a classe NEUTRO (F1 0,35) é o ponto fraco, em parte porque o próprio rótulo é ambíguo."));
children.push(B("A amostra tem cota por nota: proporções e tendência temporal no dashboard não representam a loja real."));
children.push(B("Dados em inglês, de 15 apps de produtividade, até 2020."));
children.push(B("Sem transformer pré-treinado (restrição de rede no desenvolvimento) e sem monitoramento de drift automatizado."));
children.push(B("Silhueta baixa no agrupamento: os grupos são úteis, mas o maior deles (28%) ainda mistura temas."));
children.push(H2("Trabalhos futuros"));
children.push(B("Ajustar um transformer (BERTimbau ou XLM-R) e comparar com a regressão logística no mesmo protocolo."));
children.push(B("Coletar reviews em português de apps brasileiros com google-play-scraper, sem cota por nota, e reavaliar."));
children.push(B("Rotular manualmente uma amostra de neutros para medir o teto imposto pelo rótulo derivado da nota."));
children.push(B("Trocar o K-Means por modelagem de tópicos (BERTopic/LDA) e subdividir o grupo de falhas gerais."));
children.push(B("Relatório semanal de drift (Evidently) e alerta quando uma nova versão do app passar de X% de negativos."));
children.push(H2("Lições aprendidas"));
children.push(P("Três decisões que eu repetiria e uma que faria diferente. Repetiria: fixar as metas antes de abrir o teste, o que impediu ajustar o critério ao resultado; rodar ablações simples (a de stopwords rendeu 4 pontos de F1 com uma linha de configuração); e montar o pipeline com DVC desde o início, o que transformou cada ajuste em um `dvc repro`. Faria diferente: investigar a forma de coleta do dataset antes da EDA — a queda aparente de negativos ao longo do tempo parecia um achado de negócio e era só efeito da amostragem."));

// ---------------- Referências ----------------
children.push(H1("Referências"));
[
  "CHAPMAN, P. et al. CRISP-DM 1.0: step-by-step data mining guide. SPSS, 2000.",
  "HUTTO, C. J.; GILBERT, E. VADER: a parsimonious rule-based model for sentiment analysis of social media text. In: International AAAI Conference on Weblogs and Social Media (ICWSM), 8., 2014.",
  "PEDREGOSA, F. et al. Scikit-learn: machine learning in Python. Journal of Machine Learning Research, v. 12, p. 2825–2830, 2011.",
  "HONNIBAL, M. et al. spaCy: industrial-strength natural language processing in Python. 2020. Disponível em: https://spacy.io.",
  "MANNING, C. D.; RAGHAVAN, P.; SCHÜTZE, H. Introduction to information retrieval. Cambridge University Press, 2008.",
  "RENNIE, J. D. M. et al. Tackling the poor assumptions of naive Bayes text classifiers. In: ICML, 2003.",
  "PLATT, J. Probabilistic outputs for support vector machines and comparisons to regularized likelihood methods. Advances in Large Margin Classifiers, 1999.",
  "ROUSSEEUW, P. J. Silhouettes: a graphical aid to the interpretation and validation of cluster analysis. Journal of Computational and Applied Mathematics, v. 20, p. 53–65, 1987.",
  "DEERWESTER, S. et al. Indexing by latent semantic analysis. Journal of the American Society for Information Science, v. 41, n. 6, 1990.",
  "VALKOV, V. Google Play Store reviews dataset (coletado com google-play-scraper). 2020. Espelho: github.com/kavyasri0804/Sentiment-Analysis-on-Google-Play-Store-Data-using-NLP, commit 0a25c92.",
  "OLANO, F. google-play-scraper (Python). Disponível em: https://github.com/JoMingyu/google-play-scraper.",
  "ZAHARIA, M. et al. Accelerating the machine learning lifecycle with MLflow. IEEE Data Engineering Bulletin, v. 41, n. 4, 2018.",
  "ITERATIVE. DVC — Data Version Control. Disponível em: https://dvc.org.",
].forEach((t) => children.push(P(t, { align: AlignmentType.LEFT })));

// ---------------- Apêndices ----------------
children.push(H1("Apêndices"));
children.push(H2("A. Dicionário de dados completo"));
children.push(P("O dicionário completo do arquivo bruto está na Tabela 5 (seção 2.2), com as 12 colunas originais e as colunas derivadas. O esquema do banco está na seção 7.4."));
children.push(H2("B. Figuras adicionais"));
children.push(...FIGURE("cluster_tamanhos.png", "Tamanho dos grupos de reclamação", "Falhas gerais concentram 28% dos negativos analisados; os demais grupos têm entre 4% e 13%.", 5.6));
children.push(H2("C. Resultados por dobra e grade de hiperparâmetros"));
children.push(...TABLE("F1-macro por dobra (melhor configuração de cada modelo)", "a variação entre dobras é pequena (desvio ≤ 0,022), o que dá confiança na ordenação dos modelos.",
  ["Modelo", "Melhores hiperparâmetros", "Comb.", "Dobra 1", "2", "3", "4", "5"],
  order.map((k) => [names[k], Object.entries(r[k].melhores_params).map(([a, b]) => `${a.split("__").pop()}=${b}`).join("; ") || "—",
    r[k].combinacoes_testadas, ...r[k].f1_macro_por_dobra.map((v) => n(v))]),
  [2000, 2600, 650, 880, 870, 870, 870, 898],
  { align: [AlignmentType.LEFT, AlignmentType.LEFT, ...Array(6).fill(AlignmentType.RIGHT)] }));
children.push(H2("D. Documentação da API"));
children.push(P("A especificação OpenAPI 3 completa, gerada pelo FastAPI, está em `reports/openapi.json` no repositório e pode ser navegada em http://localhost:8000/docs com a API rodando. A Tabela 17 resume os 11 endpoints."));
children.push(H2("E. Divisão de tarefas"));
children.push(P("Trabalho individual: todas as etapas (entendimento do problema, aquisição e preparação dos dados, modelagem, avaliação, pipeline, API, dashboard, testes, CI e este relatório) foram realizadas por Eduardo Luis Bremm."));

// ---------------- Checklist ----------------
children.push(H1("Checklist de autoavaliação"));
const ck = [
  ["1. Problema e CRISP-DM (10)", ["Objetivo de negócio e de mineração separados, com critério de sucesso numérico — seção 1.2", "Ciclo CRISP-DM descrito, incluindo 5 iterações — seção 1.4"]],
  ["2. Dados (15)", ["Coleta automatizada por script, com hash; fonte e licença citadas — 2.1", "Dicionário, EDA com achados e relatório de qualidade com números — 2.2 a 2.4", "Pré-processamento dentro de Pipeline, sem vazamento — 3.2"]],
  ["3. Modelagem (20)", ["Baseline + todos os algoritmos sugeridos (NB, SVM, LR, pré-treinados) — 4.1 e 4.2", "Validação cruzada estratificada e busca em grade — 4.3", "Tabela comparativa com média ± desvio — 4.4; clustering de reclamações — 4.5"]],
  ["4. Avaliação (10)", ["Teste usado uma única vez; metas atingidas, com números — 5.1", "Interpretação (coeficientes e termos por predição) e análise de erros — 5.4 e 5.5"]],
  ["5. Pipeline (15)", ["Etapas automatizadas (DVC + Makefile) e experimentos rastreados (MLflow) — 6.1 e 6.2", "Modelo versionado no Model Registry e versão gravada em cada predição — 6.2"]],
  ["6. Arquitetura (15)", ["Diagrama, API documentada, persistência e dashboard funcionando — seção 7"]],
  ["7. Reprodutibilidade (10)", ["docker compose up + dvc repro reproduzem tudo; 33 testes e CI configurado — seção 8"]],
  ["8. Apresentação (5)", ["Roteiro de demonstração pronto — 9.2 (vídeo a gravar)"]],
];
ck.forEach(([h, items]) => { children.push(H3(h)); items.forEach((t) => children.push(B("☑ " + t))); });

// ======================================================================================
const doc = new Document({
  creator: "Eduardo Luis Bremm",
  title: "ReviewLens — Relatório do Projeto de Mineração de Dados",
  styles: {
    default: { document: { run: { font: FONT, size: 22 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: FONT, size: 32, bold: true, color: ACCENT },
        paragraph: { spacing: { before: 240, after: 160 }, keepNext: true, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: FONT, size: 26, bold: true, color: ACCENT },
        paragraph: { spacing: { before: 240, after: 120 }, keepNext: true, outlineLevel: 1 } },
      { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: FONT, size: 23, bold: true, color: "333333" },
        paragraph: { spacing: { before: 180, after: 80 }, keepNext: true, outlineLevel: 2 } },
    ],
  },
  numbering: {
    config: [
      { reference: "bullets", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT,
        style: { paragraph: { indent: { left: 540, hanging: 270 } } } },
      { level: 1, format: LevelFormat.BULLET, text: "–", alignment: AlignmentType.LEFT,
        style: { paragraph: { indent: { left: 1080, hanging: 270 } } } }] },
      { reference: "steps", levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT,
        style: { paragraph: { indent: { left: 540, hanging: 360 } } } }] },
    ],
  },
  features: { updateFields: true },
  sections: [{
    properties: { page: { size: { width: 11906, height: 16838 }, margin: { top: 1134, bottom: 1134, left: 1134, right: 1134 } } },
    headers: { default: new Header({ children: [new Paragraph({ alignment: AlignmentType.RIGHT,
      children: [new TextRun({ text: "ReviewLens — Tema 9 · Mineração de Dados", font: FONT, size: 16, color: "8A8984" })] })] }) },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER,
      children: [new TextRun({ children: [PageNumber.CURRENT], font: FONT, size: 18, color: "52514E" })] })] }) },
    children,
  }],
});

Packer.toBuffer(doc).then((buf) => {
  const out = "/home/claude/report/Relatorio_ReviewLens_Tema9_EduardoBremm.docx";
  fs.writeFileSync(out, buf);
  console.log("ok", out, "figuras:", figN, "tabelas:", tabN);
});
