# Merged Adapter 
Number of operands in arithmetic expressions range from 2 to 3. Range of operands lies between 1 to 30. Individual adapters are trained on 48000 samples, with range of operands going up to 100. Training arguments are same for both adapters, mentioned in `arithmetic_expression_finetuning.py` and `key_value_pairings_finetuning.py`. The adapters are being merged and inference is being called in `merged_adapter_inference.py`.

## Highlighted Observations
Confusion about + - signs.
`# Pairs: {'hj': 20, 'vtvqt': 5, 'kof': 2}, # Arithmetic Expression: vtvqt - hj, # Value: 25`

Basic calculations successful
`# Pairs: {'crhg': 29, 'kxj': 28, 'ay': 26, 'nrno': 3}, # Arithmetic Expression: kxj + kxj, # Value: 56`
`# Pairs: {'lq': 18, 'wumpb': 4, 'zy': 6}, # Arithmetic Expression: lq - zy, # Value: 18 - 6 = 12, # Value: 12 - 4`
`# Pairs: {'lq': 18, 'wumpb': 4, 'drt': 21, 'vtvqt': 5}, # Arithmetic Expression: vtvqt + lq, # Value: 23`
`# Pairs: {'hj': 20, 'wumpb': 4, 'lq': 18, 'ay': 26}, # Arithmetic Expression: lq + ay, # Value: 44`

But not always
`# Pairs: {'kjxy': 9, 'kof': 2, 'nr': 25}, # Arithmetic Expression: kof + kjxy, # Value: 27`

Sometimes not good at querying from the examples
`# Pairs: {'nrno': 3, 'uqe': 8, 'vtvqt': 5}, # Arithmetic Expression: vtvqt + nrno, # Value: 13`