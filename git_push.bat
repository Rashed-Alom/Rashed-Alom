#!/bin/bash

git status
git add *
echo ""
echo "===================>>> *.pyc will be removed <<<==================="
git rm *.pyc -f
echo "===================>>> *.pyc removed <<<==================="
echo ""
git add -u
git status

echo ""
echo "==================>>> Enter the commit <<<==================="
set /p string=
git commit -m "%string%"

git push origin master
pause