#! /bin/bash
### BEGIN INIT INFO
# Provides:          yourapp
# Required-Start:    nginx
# Required-Stop:
# Default-Start:     2 3 4 5
# Default-Stop:      0 1 6
# Short-Description: The main django process
# Description:       The gunicorn process that receives HTTP requests
#                    from nginx
#
### END INIT INFO
#
# Author:       mle <geobi@makina-corpus.net>
#
USER=dvt
APPLOC=/opt/dvt
APPMODULE=application:application
DAEMON=gunicorn
BIND=0.0.0.0:443
PIDFILE=/opt/dvt/logs/gunicorn.pid
LOGFILE=/opt/dvt/logs/gunicorn.log
WORKERS=4
ARGS="--keyfile /opt/dvt/certs/data-validation-tool.key --certfile /opt/dvt/certs/data-validation-tool.pem --ssl-version TLSv1_2 --timeout 720"

# Source function library.
if [ -f /etc/init.d/functions ]; then
  . /etc/init.d/functions
elif [ -f /etc/rc.d/init.d/functions ] ; then
  . /etc/rc.d/init.d/functions
else
  exit 0
fi

# This is our service name
BASENAME=`basename $0`
if [ -L $0 ]; then
  BASENAME=`find $0 -name $BASENAME -printf %l`
  BASENAME=`basename $BASENAME`
fi


case "$1" in
  start)
    if [ -f $PIDFILE ] && pkill -0 -F $PIDFILE; then
      echo "$BASENAME is already running."
      echo_failure
      echo
      exit 1
    fi
    if [ -f $PIDFILE ] ; then
       rm $PIDFILE
    fi
    if ss -tulwn | grep "$BIND" &>/dev/null; then
      echo "$BIND is already in use"
      echo_failure
      echo
      exit 1
    fi
    echo -n $"Starting $BASENAME: "
    su - $USER -c "$DAEMON --daemon --bind=$BIND --pid=$PIDFILE --workers=$WORKERS --log-file=$LOGFILE --chdir $APPLOC $ARGS $APPMODULE"
    sleep 10
    RETVAL=$?
    if [ $RETVAL -eq 0 ] && [ -f $PIDFILE ] && pkill -0 -F $PIDFILE; then
      echo "$BASENAME is running"
      exit 0
    else
      echo -n "Cant find file $PIDFILE or process is not running"
      exit 1
    fi
    ;;
  stop)
    if [ -f $PIDFILE ] && pkill -0 -F $PIDFILE; then
      echo -n "Shutting down $BASENAME: "
      kill `cat $PIDFILE`
      RETVAL=$?
      [ $RETVAL -eq 0 ] && rm -f $PIDFILE
      exit 0
    else
      echo -n "Cant find file $PIDFILE or process is not running"
      exit 1
    fi
    ;;
  force-reload|restart)
    $0 stop
    $0 start
    ;;
  status)
    if [ -f $PIDFILE ] && pkill -0 -F $PIDFILE; then
      echo -n "$BASENAME is running."
      echo_success
      echo
      exit 0
    else
      echo -n "cant find pid from $PIDFILE"
      echo_failure
      echo
      exit 1
    fi
    ;;
  *)
    echo "Usage: /etc/init.d/$APPNAME {start|stop|restart|force-reload|status}"
    exit 1
    ;;
esac

exit 0