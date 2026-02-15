import os
os.environ["SPARK_VERSION"] = "3.3"   # set BEFORE importing pydeequ

import pydeequ
from pyspark.sql import SparkSession
from pydeequ.checks import Check, CheckLevel
from pydeequ.verification import VerificationSuite


def test_dataset():
    spark = (
        SparkSession.builder
        .master("local[*]")
        .appName("data-test")
        .config(
            "spark.jars.packages",
            pydeequ.deequ_maven_coord
        )
        .getOrCreate()
    )

    data = [
        ("img1.jpg", "cat", 224, 224),
        ("img2.jpg", "dog", 224, 224),
        ("img3.jpg", "cat", 224, 224),
    ]
    df = spark.createDataFrame(data, ["image_path", "label", "width", "height"])

    result = (
        VerificationSuite(spark)
        .onData(df)
        .addCheck(
            Check(spark, CheckLevel.Error, "dataset validation")
            .hasSize(lambda x: x > 0)
            .isComplete("image_path")
            .isUnique("image_path")
            .isContainedIn("label", ["cat", "dog"])
            .isNonNegative("width")
            .isNonNegative("height")
        )
        .run()
    )

    spark.stop()
    assert result.status == "Success"
