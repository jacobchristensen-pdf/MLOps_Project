import os
import sys

os.environ["SPARK_VERSION"] = "3.3"
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

import pydeequ
from pyspark.sql import SparkSession
from pydeequ.checks import Check, CheckLevel
from pydeequ.verification import VerificationSuite


def test_dataset():
    # Spark session for deequ
    spark = (
        SparkSession.builder
        .master("local[*]")
        .appName("data-test")
        .config("spark.jars.packages", pydeequ.DEEQU_MAVEN_COORD)
        .getOrCreate()
    )

    # Dummy dataset - has to be changed to work with data infrastructure
    data = [
        ("img1.jpg", "cat", 224, 224),
        ("img2.jpg", "dog", 224, 224),
        ("img3.jpg", "cat", 224, 224),
    ]
    # Spark dataframe for data validation
    df = spark.createDataFrame(data, ["image_path", "label", "width", "height"])

    result = (
        VerificationSuite(spark)
        .onData(df)
        .addCheck(
            Check(spark, CheckLevel.Error, "dataset validation")
            .hasSize(lambda x: x > 0)               # Data must have at least 1 row/image
            .isComplete("image_path")               # Each image path has to be available
            .isUnique("image_path")                 # No duplicates
            .isContainedIn("label", ["cat", "dog"]) # Must have either "cat" or "dog" labels
            .isNonNegative("width")                 # Image with must be positive
            .isNonNegative("height")                # Image height must be positive
        )
        .run()
    )

    spark.stop()
    assert result.status == "Success"
