The model is run using the [AWD LSTM language model](https://github.com/salesforce/awd-lstm-lm)

To run the model with `main.py` with bias regularization -  
 + Set the value of the the booleans `--bias_reg_encoder` and  `--bias_reg_decoder` as True
 + Set the value of `--bias_reg_en_factor`  and  `--bias_reg_de_factor`
 + Input the `--gender_pair_file` from the repo if you are using PTB, wikitext-2 or CNN/Dailymail. If you are using a different corpus, you will need to create your own based on the corpus.

To run the bias scores on a text corpus -
+ `python fixed_context_bias_score.py [input_dir] [output_dir] --gender_pair_file [penn/wiki] -w [window] -n [num_workers]
Example:
+ `python fixed_context_bias_score.py ./generated/ ./bias_data_output/gen_m75-f25/ --gender_pair_file penn -w 10 -n 1
+ `python infinite_context_bias_score.py ./awd-lstm/data/penn/ ./bias-output/ --gender_pair_file penn -w 10 -n 1
