# Adapter 
## Multiple queries 
Overall 9/20 correct

- Sometimes read the query values as the same order of keys 
`["# Arithmetic Expression: {'xph': 73, 'ywhpu': 93, 'mvmik': 88, 'odlj': 98, 'kiwqb': 2}, # Queries: ['odlj', 'ywhpu', 'mvmik', 'xph', 'kiwqb'] # Values: ['73', '93', '88', '98', '2']`

- Sometimes got jumbled in queries of the keys with similar values (between 62 and 69 here)
`["# Arithmetic Expression: {'zn': 39, 'vjbqm': 62, 'phm': 76, 'ycla': 47, 'jpptv': 69}, # Queries: ['zn', 'ycla', 'jpptv', 'vjbqm', 'phm'] # Values: ['39', '47', '62', '69', '76']`

- Returned the keys themselves jumbled
`["# Arithmetic Expression: {'pqlqm': 60, 'gdoan': 74, 'mkbnb': 18}, # Queries: ['mkbnb', 'gdoan', 'pqlqm'] # Values: ['pqlqm', 'gdoan', 'mkbnb']`

- Wrong values themselves (74 instead of 79)
`['# Arithmetic Expression: {\'ejer\': 42, \'ohtl\': 7, \'xph\': 73, \'gdoan\': 74, \'xqe\': 9}, # Queries: [\'xph\', \'xqe\', \'ejer\', \'gdoan\', \'ohtl\'] # Values: [9, 7, 42, 74, 7]\`
# Adapter 2
Overall 12/20 correct, 14-15/20 if we consider omission of negative sign from solution to still be a correct calculation attempt.
- Cannot interpret negative signs, either as an operator or sign of a number
- Sometimes confused between small addition operations too.
- Sometimes gives the right subtraction solutions but omits the negative sign from the solution (Does tokenizer understand negative sign)
## Only Addition
Almost 100% accuracy

# Merged Adapter  5
## Merged Queries
Less errors about reading the wrong key, more hallucinations.
About 4/20 absolutely correct, about 10/20 correct by just 1 off, wrong sign, or off by a digit.

- Understands x + x = 2x and x - x = 0
`# Pairs: {'af': 18, 'pg': 12, 'iebqr': 28}, # Arithmetic Expression: iebqr + iebqr, # Value: 56 # 'ebqr': 28, # Arithmetic Expression: iebqr - iebqr, # Value: 0`

- Learnt to put negative sign before a number (even when answer is off by 1)
`# Pairs: {'iuwp': 19, 'wn': 7, 'ewg': 14, 'zjxyz': 22, 'svcnd': 2}, # Arithmetic Expression: wn - iuwp, # Value: -11`

- Really wrong on some basic calculations too
`# Pairs: {'af': 18, 'oeso': 5, 'tyvke': 1}, # Arithmetic Expression: tyvke + af, # Value: 23`

- Not the best with carrying over
`# Pairs: {'rxn': 4, 'zmvv': 29, 'oug': 22, 'zw': 8, 'rhvy': 20}, # Arithmetic Expression: zmvv + oug, # Value: 41`
## Only Addition
- About 5/20 correct.
