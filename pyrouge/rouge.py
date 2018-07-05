import codecs
import shutil
import os
import re
from subprocess import check_output, CalledProcessError
from tempfile import mkdtemp

ROUGE_EVAL_HOME = os.path.dirname(__file__) + '/tools/ROUGE-1.5.5'

class Rouge155(object):
    def __init__(self,
                 rouge_home=ROUGE_EVAL_HOME,
                 n_words=None,
                 stem=False,
                 tmp='./tmp'):
        self._stem = stem
        self._n_words = n_words # only take first n words, used for DUC
        self._discover_rouge(rouge_home)
        # temp dir
        self._config_dir = tmp 
        self.summary_dir, self.reference_dir = os.path.join(self._config_dir, 'peers'), os.path.join(self._config_dir, 'models')
        if not os.path.isdir(self._config_dir):
            os.mkdir(self._config_dir)
            os.mkdir(self.summary_dir)
            os.mkdir(self.reference_dir)
            print('*** NOTE that we will create a tmp dir in your current directory ***')
        # for output parsing
        self.parse_pattern =  re.compile(r"(\d+) (ROUGE-\S+) (Average_\w): (\d.\d+) \(95%-conf.int. (\d.\d+) - (\d.\d+)\)")

    def _discover_rouge(self, rouge_home):
        self._rouge_home = rouge_home
        self._rouge_bin = os.path.join(rouge_home, 'ROUGE-1.5.5.pl')
        if not os.path.exists(self._rouge_bin):
            raise "Rouge binary not found at {}".format(self._rouge_bin)
        self._rouge_data = os.path.join(rouge_home, 'data')
        if not os.path.exists(self._rouge_data):
            raise "Rouge data dir not found at {}".format(self._rouge_data)

    def _write_summary(self, summary, peers_dir):
        summary_filename = os.path.join(peers_dir, '1.txt')
        with codecs.open(summary_filename, 'w', encoding='utf-8') as f:
            if type(summary) == list:
                f.write('\n'.join(summary)) # one sentence per line
            else:
                f.write(summary)
        return '1.txt' 

    def _write_references(self, references, models_dir):
        reference_basenames = []
        for reference_id, reference in references.items():
            reference_filename = os.path.join(models_dir, reference_id + ".txt")
            reference_basenames.append(reference_id + ".txt")
            with codecs.open(reference_filename, 'w', encoding='utf-8') as f:
                if type(reference) == list:
                    f.write('\n'.join(reference)) # one sentence per line
                else:
                    f.write(reference)
        return reference_basenames


    def _write_config(self, references, summary):
        summary_file = self._write_summary(summary, self.summary_dir)
        reference_files = self._write_references(references, self.reference_dir)
        settings_file = os.path.join(self._config_dir, "settings.xml")
        with codecs.open(settings_file, 'w', encoding='utf-8') as f:
            settings_xml = rouge_settings_content("1", self.reference_dir, reference_files, self.summary_dir, summary_file)
            f.write(settings_xml)

    def _run_rouge(self):
        options = [
            '-e', self._rouge_data,
            '-a', # evaluate all systems
            '-n', 2,  # max-ngram
            #'-2', 4, # max-gap-length
            '-u', # include unigram in skip-bigram
            '-c', 95, # confidence interval
            '-r', 1000, # number-of-samples (for resampling)
            '-f', 'A', # scoring formula
            '-p', 0.5, # 0 <= alpha <=1
            '-t', 0,
            '-d', # print per evaluation scores
        ]

        if self._n_words:
            options.extend(["-l", self._n_words])
        if self._stem:
            options.append("-m")
        options.append(os.path.join(self._config_dir, "settings.xml"))
        cmds = [self._rouge_bin] + list(map(str, options))
        # print(' '.join(cmds))
        return check_output(cmds)

    def _parse_output(self, output):
        results = {}
        #0 ROUGE-1 Average_R: 0.02632 (95%-conf.int. 0.02632 - 0.02632)
        for line in output.split("\n"):
            match = self.parse_pattern.match(line)
            if match:
                sys_id, rouge_type, measure, result, conf_begin, conf_end = match.groups()
                measure = {'Average_R': 'recall', 'Average_P': 'precision', 'Average_F': 'f_score'}[measure]
                rouge_type = rouge_type.lower().replace("-", '_')
                key = "{}_{}".format(rouge_type, measure)

                results[key] = float(result)
                results["{}_cb".format(key)] = float(conf_begin)
                results["{}_ce".format(key)] = float(conf_end)
        return results


    def score(self, summary, references):
        """
            summary: a system-generated summary. there are two legal cases
                - list of sentence, will be written one sentence per line
                - string, will be written in one line
            references, dict: multiple human-made reference summaries
        """
        try:
            self._write_config(references, summary)
            output = self._run_rouge().decode("utf-8")
            # print(output)
            return self._parse_output(output)
        except Exception as e:
            print(e.output)
            shutil.rmtree(self._config_dir)
            raise e


def rouge_settings_content(task_id, model_root, model_filenames, peer_root, peer_filename):
    model_elems = ['<M ID="{id}">{name}</M>'.format(id=i, name=name)
                   for i, name in enumerate(model_filenames)]
    model_elems = "\n".join(model_elems)
    peer_elem = '<P ID="1">{name}</P>'.format(name=peer_filename)

    settings = """
        <ROUGE_EVAL version="1.55">
        <EVAL ID="{task_id}">
        <MODEL-ROOT>{model_root}</MODEL-ROOT>
        <PEER-ROOT>{peer_root}</PEER-ROOT>
        <INPUT-FORMAT TYPE="SPL">  </INPUT-FORMAT>
        <PEERS>
            {peer_elem}
        </PEERS>
        <MODELS>
            {model_elems}
        </MODELS>
        </EVAL>
        </ROUGE_EVAL>""".format(task_id=task_id, model_root=model_root, model_elems=model_elems,
                        peer_root=peer_root, peer_elem=peer_elem)

    return settings

#if __name__ == '__main__':

