News Labeling with LLM

To construct the zero shot:

News File (name?) (Zero Shot Training News File)
pairs of timestamps & news (41 examples)

Trading File - (Zero Shot Training Currency File)
pairs of timestamps & prices

A file that contains the command that you used to generate the following

look something like: python3 news\ label.py --fx_file=data/2025\ jan\ exchnage\ rate.xlsx --news_file=data/test\ data.xlsx

(really ought to also have a --output_file="Zero Shot Training File" argument)

Combine the News File and the Trading File and threshold (news label.py)
-> Zero Shot Training File
news + positive/negative

To construct the test of the zero shot:

Zero Shot Training File - from above
New Set of News (Test News File) - pairs of timestamps & news
New Trading File - (Test Currency File) pairs of timestamps & prices


Another program that uses LLM to guess (LLM_Deepseek_labeling.py)

Eventually, a single file that just executes a bunch of commands, and all the temporary files are there and neatly labeled