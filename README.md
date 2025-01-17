# Improve-Funding-Mechanism-for-Ethereum
Given a pair of repositories (𝐴, 𝐵), your model must output a single value 𝑤(𝐴) between 0 and 1 (the “weight” for repository 𝐴) that reflects their relative share. The weight for 𝐵 will be 1−𝑤(𝐴). For example, if projects 𝐴 and 𝐵 received $100 funding in total, and 𝐴 received $45, then 𝑤(𝐴) = 0.45 and 𝑤(𝐵) = 0.55.
This problem can be formulated as a **supervised regression** task: Predict the relative funding received between any two open-source repositories based on their past funding data and dependencies between projects. As training, we have provided ~20,000 rows on the relative funding received by projects from 2019 to 2023. Your task? Give the expected relative funding on ~5000 rows on relative amounts from projects in funding rounds in 2024.
