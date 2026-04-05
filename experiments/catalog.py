CHAPTERS = [
    {
        "title": "Chapter 2: We Are All Just Numbers Here ...",
        "summary": "Linear decision boundaries, the perceptron rule, and first intuition for learning.",
        "experiments": [
            {
                "slug": "perceptron-intro",
                "title": "Experiment 1: Perceptron Playground",
                "status": "Interactive",
                "description": "Train a perceptron on synthetic 2D data and inspect decision boundaries epoch by epoch.",
                "template": "perceptron.html",
            },
            
        ],
    },
    {
        "title": "Chapter 3: The Bottom of the Bowl",
        "summary": "Least Mean Squares objective, update rules, and convergence behavior.",
        "experiments": [
            {
                "slug": "gradient-decent",
                "title": "Gradient Descent Visualizer",
                "status": "Interactive",
                "description": "Classic gradient descent on z = x^2 + y^2 with a visible optimization path.",
                "template": "gradient.html",
            },
            {
                "slug": "lms-filter",
                "title": "Experiment 2: ADALINE Speech Denoiser",
                "status": "Interactive",
                "description": "Use an adaptive linear neuron to learn a clean speech signal from a noisy recording and inspect how the error shrinks.",
                "template": "lms.html",
            }
        ],
    },

    {
        "title": "Chapter 4: In All Probability",
        "summary": "Understand Bayes rule and the montey hall problem",
        "experiments": [
            {
                "slug": "monty-hall",
                "title": "The Monty Hall Problem",
                "status": "Statistics",
                "description": "Learn how bayes rule proves an none intuitive winning strategy for winning a car instead of a goat.",
                "template": "monty.html",
            },
           
        ],
    },
    {
        "title": "Chapter 5: Birds of Feather",
        "summary": "Understand Bayes rule, the montey hall problem and more",
        "experiments": [
            {
                "slug": "k-nearest-neighbors",
                "title": "Experiment 3. k-NN Decision Boundary Lab",
                "status": "Interactive",
                "description": "Control k, noise, and sample count to see how nearest-neighbor classification bends the boundary.",
                "template": "knearest.html",
            }
           
        ],
    },
    {
        "title": "Chapter 6: There's Magic in Them Matrices",
        "summary": "Covariance, eigenvectors, and low-dimensional views of high-dimensional data.",
        "experiments": [
            {
                "slug": "iris-pca",
                "title": "Experiment 1: Iris PCA Explorer",
                "status": "Interactive",
                "description": "Center the Iris dataset, project it onto principal components, and watch how species separate in 2D or 3D.",
                "template": "iris.html",
            },
            {
                "slug": "hopfield-5-vs-8",
                "title": "Experiment 2: Hopfield Memory (5 vs 8)",
                "status": "Interactive",
                "description": "Store MNIST digits 5 and 8 with Hebbian learning, add noise, and watch asynchronous recall sweep states.",
                "template": "hopfield.html",
            },
        ],
    },
    
]

EXPERIMENT_INDEX = {
    experiment["slug"]: {
        "chapter_title": chapter["title"],
        **experiment,
    }
    for chapter in CHAPTERS
    for experiment in chapter["experiments"]
}
