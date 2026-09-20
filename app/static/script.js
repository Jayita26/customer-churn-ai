const form = document.getElementById("churnForm");

form.addEventListener("submit", async function(event) {

    event.preventDefault();

    const customerData = {

        gender:
            document.getElementById("gender").value,

        SeniorCitizen:
            parseInt(
                document.getElementById("SeniorCitizen").value
            ),

        Partner:
            document.getElementById("Partner").value,

        Dependents:
            document.getElementById("Dependents").value,

        tenure:
            parseInt(
                document.getElementById("tenure").value
            ),

        PhoneService:
            document.getElementById("PhoneService").value,

        MultipleLines:
            document.getElementById("MultipleLines").value,

        InternetService:
            document.getElementById("InternetService").value,

        OnlineSecurity:
            document.getElementById("OnlineSecurity").value,

        OnlineBackup:
            document.getElementById("OnlineBackup").value,

        DeviceProtection:
            document.getElementById("DeviceProtection").value,

        TechSupport:
            document.getElementById("TechSupport").value,

        StreamingTV:
            document.getElementById("StreamingTV").value,

        StreamingMovies:
            document.getElementById("StreamingMovies").value,

        Contract:
            document.getElementById("Contract").value,

        PaperlessBilling:
            document.getElementById("PaperlessBilling").value,

        PaymentMethod:
            document.getElementById("PaymentMethod").value,

        MonthlyCharges:
            parseFloat(
                document.getElementById("MonthlyCharges").value
            ),

        TotalCharges:
            parseFloat(
                document.getElementById("TotalCharges").value
            )
    };


    try {

        const response = await fetch("/predict", {

            method: "POST",

            headers: {
                "Content-Type": "application/json"
            },

            body: JSON.stringify(customerData)

        });


        if (!response.ok) {

            throw new Error(
                "Prediction request failed"
            );

        }


        const result = await response.json();


        document.getElementById("prediction").textContent =
            result.prediction;


        document.getElementById("probability").textContent =
            result.churn_probability_percent + "%";


        document
            .getElementById("result")
            .classList
            .remove("hidden");


    } catch (error) {

        alert(
            "Error connecting to the prediction API."
        );

        console.error(error);

    }

});