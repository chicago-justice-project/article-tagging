import sys
from .tag import Tagger

"""
A command line interface to the automatic article tagger.
Run with `python -m tagnews.crimetype.cli`
"""

if __name__ == '__main__':
    tagger = Tagger()

    if len(sys.argv) == 1:
        print(('Go ahead and start typing.'
               '\nIf you are on a UNIX machine, hit ctrl-d when done.'
               '\nIf you are on a Windows machine, hit ctrl-Z and then Enter when done.'))
        s = sys.stdin.read()
        preds = tagger.tagtext_proba(s)
        preds = preds.sort_values(ascending=False)
        for tag, prob in zip(preds.index, preds.values):
            print('{: >5}, {:.9f}'.format(tag, prob))
    else:
        if sys.argv[1] in ['-h', '--help']:
            h = 'python -m tagnews.crimetype.tag [filename [filename [...]]]\n'
            h += '\n'
            h += 'If no filenames are provided, read and tag from stdin.\n'
            h += '(Use ctrl-d to stop inputting to stdin.)\n'
            h += '\n'
            h += 'Otherwise, tag all filenames, outputting the tags as a CSV\n'
            h += 'to the file <filename>.tagged.'
            print(h)
            quit()
        for filename in sys.argv[1:]:
            with open(filename) as f_in:
                preds = tagger.tagtext_proba(f_in.read())
            preds = preds.sort_values(ascending=False)
            with open(filename + '.tagged', 'w') as f_out:
                for tag, prob in zip(preds.index, preds.values):
                    f_out.write('{: >5}, {:.9f}\n'.format(tag, prob))
