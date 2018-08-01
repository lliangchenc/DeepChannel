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
                 n_bytes=None,
                 stem=False,
                 tmp='./tmp'):
        self._stem = stem
        self._n_bytes = n_bytes # only take first n words, used for DUC
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
        self.parse_pattern = re.compile(r"(\d+) (ROUGE-\S+) (Average_\w): (\d.\d+) \(95%-conf.int. (\d.\d+) - (\d.\d+)\)")

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

        if self._n_bytes:
            options.extend(["-b", self._n_bytes])
        if self._stem:
            options.append("-m")
        options.append(os.path.join(self._config_dir, "settings.xml"))
        cmds = [self._rouge_bin] + list(map(str, options))
        output = check_output(cmds).decode("utf-8")
        # parse output
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


    def rouge_settings(self, model_root, peer_root, pair_filenames):
        """
        'model' means ground truth, while 'peer' means generated summary
        pair_filenames : [(peer_filename, [model_filename1, model_filename2, ...]), ...]
                or [(peer_filename, model_filename), ...]
        """
        settings_file = os.path.join(self._config_dir, "settings.xml")
        settings = '<ROUGE_EVAL version="1.55">'
        for task_id, pair in enumerate(pair_filenames):
            peer_filename, model_filenames = pair
            if isinstance(model_filenames, str):
                model_filenames = [model_filenames]
            model_elems = ['<M ID="{id}">{name}</M>'.format(id=i, name=name)
                       for i, name in enumerate(model_filenames)]
            model_elems = "\n".join(model_elems)
            peer_elem = '<P ID="1">{name}</P>'.format(name=peer_filename)

            settings += """
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
                </EVAL>""".format(task_id=task_id, model_root=model_root, model_elems=model_elems,
                            peer_root=peer_root, peer_elem=peer_elem)
        settings += '</ROUGE_EVAL>'
        
        with codecs.open(settings_file, 'w', encoding='utf-8') as f:
            f.write(settings)


    def score(self, summary, references):
        """
        Evaluate a pair of summary and references. Support multiple references such as DUC.
            summary: a system-generated summary. there are two legal cases
                - list of sentence, will be written one sentence per line
                - string, will be written in one line
            references, dict: multiple human-made reference summaries
        """
        try:
            summary_file = self._write_summary(summary, self.summary_dir)
            reference_files = self._write_references(references, self.reference_dir)
            self.rouge_settings(self.reference_dir, self.summary_dir, [(summary_file,reference_files)])
            return self._run_rouge()
        except Exception as e:
            print(e.output)
            shutil.rmtree(self._config_dir)
            raise e


    def evaluate_folder(self, summary_folder, reference_folder, summary_file_suffix='_decoded.txt', reference_file_suffix='_reference.txt'):
        """
        Evaluate multiple pairs which have been written into files.
        Each file in summary_folder must have a corresponding file in reference_folder. That is, there must be <id>[reference_file_suffix] in reference_folder if there is a summary named <id>[summary_file_suffix].
        """
        try:
            pairs = []
            for f in os.listdir(summary_folder):
                id = f[:-len(summary_file_suffix)]
                pairs.append((f, id+reference_file_suffix)) # file name of (summary, reference) pairs
            self.rouge_settings(reference_folder, summary_folder, pairs)
            return self._run_rouge()
        except Exception as e:
            print(e.output)
            shutil.rmtree(self._config_dir)
            raise e

    def evaluate_folder_macro_average(self, summary_folder, reference_folder, summary_file_suffix='_decoded.txt', reference_file_suffix='_reference.txt'):
        """
        Evaluate multiple pairs which have been written into files.
        Each file in summary_folder must have a corresponding file in reference_folder. That is, there must be <id>[reference_file_suffix] in reference_folder if there is a summary named <id>[summary_file_suffix].
        Call self.score for each pair, and then average results of all pairs.
        """
        total_score = None
        count = 0
        for f in os.listdir(summary_folder):
            id = f[:-len(summary_file_suffix)]
            summary_file = os.path.join(summary_folder, f)
            reference_file = os.path.join(reference_folder, id+reference_file_suffix)
            summary = open(summary_file).readlines()
            references = {'A': open(reference_file).readlines()}
            one_score = self.score(summary, references)
            count += 1
            if total_score is None:
                total_score = one_score
            else:
                for k in one_score:
                    total_score[k] += one_score[k]
        for k in total_score:
            total_score[k] /= count
        return total_score



#if __name__ == '__main__':

