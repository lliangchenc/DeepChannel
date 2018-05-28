
		DUC 2007 Main Task: Evaluation Results
		======================================


This is the README file for the manual and automatic evaluation that
was performed at NIST for summaries in the DUC 2007 main task.

Ten NIST assessors wrote summaries for the 45 topics in the DUC 2007
main task.  Each topic had 4 human summaries.  The human summarizer
IDs are A-J.

Two baseline summarizers were included in the evaluation:

    Baseline 1 (summarizer ID = 1): return all the leading sentences
    (up to 250 words) in the <TEXT> field of the most recent document.

    Baseline 2 (summarizer ID = 2): CLASSY04, an automatic summarizer
    that ignores the topic narrative but that had the highest mean SEE
    coverage score in Task 2 of DUC 2004, a multi-document
    summarization task. (For a description of the system, see
    <http://duc.nist.gov/pubs/2004papers/ida.conroy.ps>.)

NIST received submissions from 30 different participants for the main
task.  The participants' summarizer IDs are 3-32.

All summaries were truncated to 250 words before being evaluated.
Summaries are saved one per file, using the following naming
convention: <topic>.M.250.<topic_selectorID>.<summarizerID>



Manual Evaluation
-----------------

Linguistic Quality: NIST assessors judged each summary for linguistic
quality. Five Quality Questions were used:
	1. Grammaticality
	2. Non-redundancy
	3. Referential clarity
	4. Focus
	5. Structure and Coherence
(See the DUC2007 Task Description for a detailed description of each
quality question.)

Responsiveness: NIST assessors assigned a content responsiveness score
to each of the automatic and human summaries.  The content score is an
integer between 1 (very poor) and 5 (very good) and is based on the
amount of information in the summary that helps to satisfy the
information need expressed in the topic narrative.

Files under manual/:
   peers/			human and automatic summaries (original format)
   linguistic_quality.table	scores for linguistic quality questions
   content.table		scores for content responsiveness
   avg_content			average of content score, by summarizer ID



ROUGE
-----

ROUGE-2 and ROUGE-SU4 scores were computed by running ROUGE-1.5.5 with
stemming but no removal of stopwords.  The input file implemented
jackknifing so that scores of systems and humans could be compared.

ROUGE run-time parameters:
	ROUGE-1.5.5.pl -n 4 -w 1.2 -m  -2 4 -u -c 95 -r 1000 -f A -p 0.5 -t 0 -a -d rougejk.in 

Files under ROUGE/:
   models/		sentence-segmented human summaries
   peers/		sentence-segmented human and automatic summaries
   rougejk.in		input file to ROUGE-1.5.5
   rougejk.m.out	output of ROUGE-1.5.5
   rouge2.jk.m.avg	average ROUGE-2 recall, by summarizer ID
   rougeSU4.jk.m.avg	average ROUGE-SU4 recall, by summarizer ID



Basic Elements
--------------

Basic Elements (BE) scores were computed by first using the tools in
BE-1.1 to extract BE-F from each sentence-segmented <summary>:
   bebreakMinipar.pl -p $MINIPATH <summary>

The BE-F were then matched by running ROUGE-1.5.5 with stemming, using
the Head-Modifier (HM) matching criterion.  The input file to
ROUGE-1.5.5 implemented jackknifing so that scores of systems and
humans could be compared.  

BE run-time parameters:
	ROUGE-1.5.5.pl -m -a -d -3 HM simplejk.in  

Files under BE/:
   BEmodels/			BEs from human summaries
   BEpeers/			BEs from human and automatic summaries
   simplejk.in			input file to ROUGE-1.5.5
   simplejk.m.hm.out		output of ROUGE-1.5.5
   simple.jk.m.hm.avg		average BE recall, by summarizer ID
