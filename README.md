## Dependencies
Python 3.6>, MongoDB
From PIP: pycrfsuite, arrow..... (can't recall all of them for now.)

## File Descriptions (./)
1. ``entry.py``: An entry file for an end user. You may use ``python entry.py --help`` to see more commands.
1. ``./scrabble/scrabble.py``: Main class file.
2. ``./scrabble/char2ir.py``: Character to intermediate representation (IR or Tags) mapping with CRF.
3. ``./scrabble/ir2tagsets.py``: IR to TagSets learning and iteration functions using MLP.

## Data Schema for MongoDB (``scrabble/data_model.py``)
All raw metadata and labeled metadata are stored in MongoDB. Please refer the comments in the above file for the schema.

## How to use?
### Configuration options
1. ``python entry.py --help`` will show below.
2. Essential Parameters
    - ``-task``: Types of the learning task. One of ``scrabble``, ``char2ir`` and ``ir2tagsets``. ``scrabble`` tests the entire model and each of the others does for each step (1st and 2nd.)
    - ``-bl``: Building list: list of source building names deliminated by comma (e.g., -bl ebu3b,bml).
    - ``-nl``: Sample number list: list of sample numbers per building. The order should be same as bl. (e.g., -nl 200,1)
    - ``-t``: Target building name: Name of the target building. (e.g., -t bml)
    - ``-inc``: New examples to ask a domain expert in each iteration. (default: ``10``)
    - ``-iter``: Total number of iterations. (default: ``20``)
3. Other Parameters (Mainly model configurations)
    - ``-ct``: 2nd stage classifier type. (default: ``MLP``)
    - ``-c``: Whether to use clustering for random selection or not (default: ``true``)
    - ``-neg``: Use sample augmentation using negative sample synthesis. (default: ``true``)
    - ``-ub``: Whether to use Brick when learning. (default: ``true``).
    - ``-crfqs``: Active learning query strategy for ``char2ir`` (default: ``confidence``).
    - ``-entqs``: Active learning query strategy for ``ir2tagsets`` (default: ``phrase_util``).

### Preprocessing (don't do it until you understand what they do.)
1. Generate ground truth (from schema_map at Dropbox.)
2. Tokenize sentences with predefined rules (`sentence_normalizer.ipynb`)
3. Label them with rules and experts (`label_with_experts.ipynb`)
4. Generate tokenized labels. (`conv_word_labels_to_char_labels.ipynb`)
5. Generate ground truth file (`ground_truth_gen.py <building>`)

### Result Intepretation
1. Look at ``ResultHistory`` in ``scrabble/data_model.py`` for more information.
2. TODO: Need to clean this up.
