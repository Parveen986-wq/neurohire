"""Skill taxonomy and job role profiles used for matching, training and recommendations."""

ROLES: dict[str, list[str]] = {
    "Data Scientist": ["Python", "Machine Learning", "Statistics", "Pandas", "NumPy", "Scikit-learn",
                       "SQL", "Deep Learning", "TensorFlow", "Data Visualization", "NLP"],
    "Machine Learning Engineer": ["Python", "TensorFlow", "PyTorch", "Deep Learning", "MLOps", "Docker",
                                  "Kubernetes", "Scikit-learn", "AWS", "Machine Learning", "Git"],
    "Frontend Developer": ["JavaScript", "TypeScript", "React", "HTML", "CSS", "Tailwind CSS",
                           "Redux", "Next.js", "Git", "REST API", "Figma"],
    "Backend Developer": ["Python", "Java", "Node.js", "SQL", "PostgreSQL", "REST API", "Docker",
                          "FastAPI", "Django", "Microservices", "Git", "Redis"],
    "Full Stack Developer": ["JavaScript", "TypeScript", "React", "Node.js", "SQL", "MongoDB",
                             "REST API", "Docker", "Git", "HTML", "CSS"],
    "DevOps Engineer": ["Docker", "Kubernetes", "AWS", "Linux", "CI/CD", "Terraform", "Jenkins",
                        "Git", "Bash", "Azure", "Monitoring"],
    "Data Analyst": ["SQL", "Excel", "Power BI", "Tableau", "Python", "Pandas", "Statistics",
                     "Data Visualization", "Communication"],
    "Android Developer": ["Kotlin", "Java", "Android", "Firebase", "REST API", "Git", "Jetpack Compose"],
}

SOFT_SKILLS = ["Communication", "Leadership", "Teamwork", "Problem Solving", "Time Management"]

SKILLS = sorted({s for v in ROLES.values() for s in v} | set(SOFT_SKILLS) | {
    "C", "C++", "Go", "Rust", "Spark", "Hadoop", "GraphQL", "Vue", "Angular", "Flask",
    "OpenCV", "Keras", "spaCy", "Transformers", "LLM", "GCP", "MySQL", "Agile", "Scrum",
})

LEARNING = {
    "Python": "Python for Everybody (Coursera)",
    "Machine Learning": "Andrew Ng - Machine Learning Specialization",
    "Deep Learning": "DeepLearning.AI Deep Learning Specialization",
    "TensorFlow": "TensorFlow Developer Certificate",
    "PyTorch": "PyTorch official tutorials",
    "Docker": "Docker Mastery (Udemy)",
    "Kubernetes": "CKAD certification path",
    "AWS": "AWS Certified Cloud Practitioner",
    "React": "react.dev official docs + build 3 projects",
    "SQL": "SQLBolt + LeetCode SQL 50",
    "TypeScript": "TypeScript Handbook",
    "NLP": "Hugging Face NLP Course",
}
