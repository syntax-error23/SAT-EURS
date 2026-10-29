# EURS @ MaCVi WACV 2027 – factsheet template

LaTeX template for the factsheet every EURS team sends after the testing phase.
Based on the NTIRE 2024 factsheet format (team details, global method description,
technical details, other details), adapted to the two EURS tracks.

## Files
- `main.tex` – the factsheet (instructions section at the top, red `[...]` placeholders to fill in)
- `ref.bib` – references (replace the example entry)
- `images/` – put your figures here
- `EURS_factsheet_template.pdf` – the compiled empty template

## Fill in and build
1. Fill every red placeholder (`\todo{...}`) and delete the example text. Skip Section 3 or 4 if you did not enter that track.
2. Set `\setboolean{showinstructions}{false}` near the top of `main.tex` to drop the instructions page from your PDF.
3. Build with `latexmk -pdf main.tex` (it runs LaTeX and BibTeX as often as needed). A single plain `pdflatex` pass leaves "??" in the cross-references and "[?]" in citations.

## Submit (deadline: October 28, 2026, 23:59 AoE; late factsheets are not considered)
E-mail the compiled **PDF** and a **zip of the LaTeX sources** (`main.tex`, `ref.bib`, `images/`) to:

- Nikhil Akalwadi – nikhil.akalwadi@kletech.ac.in
- Sampada Malagi – sampadamalagi12@gmail.com
- Sneha Varur – sneha.varur@kletech.ac.in
- Chaitra Desai – chaitra.desai@manipal.edu

with **all team members in cc**. Subject: `MaCVi@WACV 2027 EURS Challenge - TEAM_NAME`.
The e-mail body lists the challenge name, team name, leader and e-mail, members, macvi.org user IDs,
the best-scoring testing-phase submission ID and its user ID per track, all team members' user IDs, the link to code and weights, and the attachments.

Make the zip: `zip -r factsheet_TEAM_NAME.zip main.tex ref.bib images`

## License

Our text and layout are released under the MIT License (see `LICENSE`). The structure follows the NTIRE 2024 factsheet format, as noted above.

