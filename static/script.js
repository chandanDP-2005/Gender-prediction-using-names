const nameInput = document.getElementById("name");

const form = document.getElementById("form");

const waiting = document.getElementById("waiting");

const result = document.getElementById("result");

const analysis = document.getElementById("analysis");

const status = document.getElementById("status");


/* Example buttons */

document
.querySelectorAll(".examples button")
.forEach(button => {

    button.addEventListener("click", () => {

        nameInput.value =
            button.textContent;

        predictName(
            button.textContent
        );

    });

});


/* Clear */

document
.getElementById("clear")
.addEventListener("click", () => {

    nameInput.value = "";

    nameInput.focus();

});


/* Submit */

form.addEventListener(
    "submit",
    function(event) {

        event.preventDefault();

        const name =
            nameInput.value.trim();

        if (!name) {

            alert(
                "Please enter a name."
            );

            return;

        }

        predictName(name);

    }
);


/* Prediction */

async function predictName(name) {

    status.textContent =
        "ANALYZING";

    try {

        const response =
            await fetch(
                "/predict",
                {
                    method: "POST",

                    headers: {
                        "Content-Type":
                            "application/json"
                    },

                    body:
                        JSON.stringify({
                            name: name
                        })
                }
            );


        const data =
            await response.json();


        if (!data.success) {

            alert(data.error);

            return;

        }


        const r =
            data.result;


        /* Hide waiting */

        waiting.classList.add(
            "hidden"
        );

        result.classList.remove(
            "hidden"
        );

        analysis.classList.remove(
            "hidden"
        );


        status.textContent =
            "COMPLETE";


        /* Gender */

        document.getElementById(
            "gender"
        ).textContent =
            r.prediction;


        document.getElementById(
            "genderIcon"
        ).textContent =
            r.prediction === "Male"
                ? "M"
                : "F";


        /* Confidence */

        animateNumber(
            "confidence",
            r.confidence
        );


        /* Probability */

        animateNumber(
            "male",
            r.probabilities.Male
        );


        animateNumber(
            "female",
            r.probabilities.Female
        );


        setTimeout(() => {

            document.getElementById(
                "mainProgress"
            ).style.width =
                r.confidence + "%";


            document.getElementById(
                "maleBar"
            ).style.width =
                r.probabilities.Male + "%";


            document.getElementById(
                "femaleBar"
            ).style.width =
                r.probabilities.Female + "%";

        }, 100);


        /* NLP features */

        document.getElementById(
            "length"
        ).textContent =
            r.features.length;


        document.getElementById(
            "first"
        ).textContent =
            r.features.first_character;


        document.getElementById(
            "last"
        ).textContent =
            r.features.last_character;


        document.getElementById(
            "normalized"
        ).textContent =
            r.normalized;


        createTokens(
            "prefixes",
            r.features.prefixes
        );


        createTokens(
            "suffixes",
            r.features.suffixes
        );


        createTokens(
            "bigrams",
            r.features.bigrams
        );


        createTokens(
            "trigrams",
            r.features.trigrams
        );


        analysis.scrollIntoView({
            behavior: "smooth"
        });


    }

    catch(error) {

        console.error(error);

        alert(
            "Server error. Make sure Flask is running."
        );

    }

}


/* Number animation */

function animateNumber(
    id,
    target
) {

    const element =
        document.getElementById(id);

    let start = 0;

    const duration = 700;

    const startTime =
        performance.now();


    function update(time) {

        const progress =
            Math.min(
                (time - startTime)
                / duration,
                1
            );


        const value =
            start +
            (target - start)
            * progress;


        element.textContent =
            value.toFixed(1) + "%";


        if (progress < 1) {

            requestAnimationFrame(
                update
            );

        }

    }


    requestAnimationFrame(
        update
    );

}


/* Create NLP tokens */

function createTokens(
    id,
    values
) {

    const container =
        document.getElementById(id);

    container.innerHTML = "";


    values.forEach(
        (value, index) => {

            const span =
                document.createElement(
                    "span"
                );


            span.className =
                "token";


            span.textContent =
                value;


            span.style.animationDelay =
                (index * 50) + "ms";


            container.appendChild(
                span
            );

        }
    );

}


/* Load training statistics */

async function loadStats() {

    try {

        const response =
            await fetch("/stats");

        const data =
            await response.json();


        document.getElementById(
            "total"
        ).textContent =
            data.total;


        document.getElementById(
            "maleCount"
        ).textContent =
            data.male;


        document.getElementById(
            "femaleCount"
        ).textContent =
            data.female;

    }

    catch(error) {

        console.log(
            "Statistics unavailable"
        );

    }

}


loadStats();