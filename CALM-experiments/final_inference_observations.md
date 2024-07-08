# Adapter 1
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
## No formatting, multiple queries
Shabby results, almost none accuracy
## No formatting, single queries
Close to 90% accuracy
# Adapter 2
Overall 12/20 correct, 14-15/20 if we consider omission of negative sign from solution to still be a correct calculation attempt.
- Cannot interpret negative signs, either as an operator or sign of a number
- Sometimes confused between small addition operations too.
- Sometimes gives the right subtraction solutions but omits the negative sign from the solution (Does tokenizer understand negative sign)
## Only Addition
Almost 100% accuracy
## Without prompt formatting
Almost 100% accuracy. ('Almost' because of entries like this:`'7 + 21 = ?'`)
Some interesting results:
- `'25 + 4 = 29\nSarah is 29 years old. She is 25 years younger than her mother. How old is her mother?\nA. 54 years old`
- `'8 + 14 = ?\nWhat is the relationship between 8 and 14?\nWhat is the relationship between 8 and 14? 8 is one more than 7. 14 is one less than 15.\n8 + 14 = 22.`

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
## No prompt format 18/26
69% accuracy, 73% for one-off errors without reading issues
4 oneoff
One off accuracy still persists
- `{'ryz': 13, 'yowq': 18, 'fwzp': 2, 'ekffy': 20}, ekffy + yowq = 39`
Some interestingly wrong results
- `{'bjob': 22, 'zyjd': 4, 'amatg': 7, 'tcj': 15}, bjob + bjob = 44, bjob + 44 = 92, bjob + 92 = 184, bjob + 184 = 366,`
Good at key pair readings but fumbling at calculation sometimes
- `{'mtwl': 14, 'ekffy': 20, 'gbnp': 3}, gbnp + mtwl = 54, gbnp = 54 - mtwl, gbnp = 54 - 14 = 38`
