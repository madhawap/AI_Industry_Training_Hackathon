ok [medium]   6.0s ops=1 2/2  Which single day had the most AFR articles published, and how ma
   [hard  ]  11.5s ops=2 0/3  How many AFR articles mention 'bank' in their headline or text, 
       missed: ['20', '72 characters', 'Computershare revenue falls as record low interest rates squeeze margins']
       answer: 13 AFR articles mention 'bank' in their headline or text. The longest headline cannot be determined because the retrieval query returned 0 articles, with a warning that n
ok [easy  ]   8.4s ops=1 5/5  Which ASX ticker symbols are present in this dataset, and how ma
ok [medium]   7.1s ops=1 2/2  Which ASX company has the highest average closing price across t
   [medium]   7.4s ops=1 2/4  What was CBA's largest single-day percentage gain in closing pri
       missed: ['82.55', '85.63']
       answer: CBA's largest single-day percentage gain in closing price was 3.7311%, which occurred on 2019-01-29.
   [hard  ]   5.4s ops=1 3/4  What is the longest gap, in days, between two consecutive RBA ra
       missed: ['4.1']
       answer: The longest gap between two consecutive RBA rate cuts is 1568 days, spanning from 2020-11-04 to 2025-02-19.
ok [hard  ]   5.5s ops=1 3/3  What is the longest gap, in days, between two consecutive RBA ra
   [medium]   7.0s ops=1 2/3  Using relevance-ranked full-text search, which AFR article is mo
       missed: ['2015-01-13']
       answer: The most relevant AFR article is "Mineral Resources strikes iron ore port agreement with Pilbara partners," and its BM25 score is 2.90.
   [medium] 175.8s ops=2 2/3  How many RBA decision records show a cash rate target of either 
       missed: ['42']
       answer: The provided results do not contain the number of RBA decision records for cash rate targets of 1.5% or 2.0%, as they only report the record counts for the highest rate (
   [medium]   7.1s ops=1 2/3  Across how many AFR articles does the headline mention 'RBA', an
       missed: ['5']
       answer: The headline mentions 'RBA' in 4 AFR articles, spanning from 2015-01-07 to 2015-03-03.
ok [hard  ]   8.8s ops=1 6/6  What are BHP's three highest single-day trading volumes in this 
   [hard  ]  20.0s ops=3 3/4  What was the RBA cash rate target in effect on the day BHP recor
       missed: ['37.09']
       answer: The RBA cash rate target in effect on 2021-08-26 was 0.1%, having been set on 2021-08-04 with a change of 0.0 percentage points. The verification results could not determ
   [hard  ]  23.9s ops=4 2/5  On the day of BHP's largest single-day percentage decline, what 
       missed: ['-5.09%', '2020-03-04', '0.5']
       answer: On 2020-03-03, the RBA cash rate target in effect was 0.75% (set on 2020-02-05), and no RBA rate decision took effect within the following two days.
ok [medium]   8.2s ops=1 3/3  Which ASX tickers have a higher average closing price than CBA.A
   [medium]  13.2s ops=1 2/4  In the calendar month containing BHP's single highest trading-vo
       missed: ['18439800', '1']
       answer: BHP's single highest trading-volume day occurred on 2015-08-24, placing it in the calendar month of August 2015. The provided results do not contain data regarding RBA ra
   [medium]  13.5s ops=2 2/3  Do any AFR articles in this dataset fall within the calendar mon
       missed: ['0']
       answer: No, no AFR articles in this dataset fall within November 2020, the calendar month the RBA cash rate first hit its record low of 0.1% on 2020-11-04, as the dataset only co

==============================================================
component score   59.6/82  (72.7%)
perfect answers   10/22
latency           mean 17.0s  p95 20.0s
over 60s          1  (each loses 20% of its earned points)
no tool call      0
mean operations   1.4
wall clock        208.1s at concurrency 3