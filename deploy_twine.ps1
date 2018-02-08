if ($env:APPVEYOR_REPO_TAG -eq "true" -And $env:APPVEYOR_REPO_BRANCH -eq "master") {
    Write-Output "Deploying to PyPi"
    pip install twine
    twine upload dist\\* 
    Write-Output "Done"
} else {
    Write-Output "Skipping deployment"
}