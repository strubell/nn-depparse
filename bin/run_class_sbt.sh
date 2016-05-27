#!/bin/bash
# run_class.sh [<jvm options>] <scala object with main> [<arguments>]

args="$@"

memory=2g

PROJ_ROOT=$NNDEPPARSE_ROOT

# If the root directory is not passed as an environment variable, set it to 
# the current directory.
if [ -z "$PROJ_ROOT" ]
then
  PROJ_ROOT=`pwd`
fi

#echo "PROJ_ROOT: "$PROJ_ROOT

# Get classpath: all jar files from sbt -- either from a file CP.hack, or 
# if this file does not exist yet, from the output of a maven compile.
# TODO: there should be a nice and automated way for starting after building -
# for now, this hack attempts that, somehow.
if [ ! -f "$PROJ_ROOT/CP.hack" ]
then
 echo 'CP.hack does not exist, get class paths ...'
 cd $PROJ_ROOT
 echo -n "$PROJ_ROOT/target/scala-2.11/classes/:" > $PROJ_ROOT/CP.hack
 sbt -Dsbt.log.noformat=true "show compile:dependency-classpath" \
 | grep "List(Attributed" \
 | sed 's/.*List(Attributed(\(.*\)))/\1/' \
 | sed 's/), Attributed(/:/g' \
 >> $PROJ_ROOT/CP.hack
 cd -
 echo '... done'
fi

CP=`cat $PROJ_ROOT/CP.hack`

java -Dfile.encoding=UTF8 -cp $CP -Xmx$memory $args
#java -cp $CP -Xmx$memory $args
