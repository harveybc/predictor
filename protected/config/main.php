<?php

// uncomment the following to define a path alias
// Yii::setPathOfAlias('local','path/to/local-folder');
// This is the main Web application configuration. Any writable
// CWebApplication properties can be configured here.
return array(
    'basePath' => dirname(__FILE__) . DIRECTORY_SEPARATOR . '..',
    'name' => 'CBM',
    // preloading 'log' component
    'preload' => array('log'),
    // autoloading model and component classes
    'import' => array(
        'application.models.*',
        'application.components.*',
        'ext.giix-components.*', // giix components
        'ext.*', // giix components
        'ext.highcharts.*', // giix components
        'ext.highstock.*', // giix components
        'ext.jstree.*', // jstree components
        'ext.editor.*', // jstree components
    ),
    'modules' => array(
        // uncomment the following to enable the Gii tool

        'gii' => array(
            'class' => 'system.gii.GiiModule',
            'password' => '0ptimus',
            'generatorPaths' => array(
                'ext.giix-core',
                'ext.gtc', // Gii Template Collection// giix generators
            ),
            // If removed, Gii defaults to localhost only. Edit carefully to taste.
            'ipFilters' => array('127.0.0.1', '::1', '190.84.49.118', '10.11.0.10', '10.11.0.1'),
        ),
    ),
    // application components
    'components' => array(
        'browser' => array(
            'class' => 'application.extensions.browser.CBrowserComponent',
        ),
        'cache' => array(
            'class' => 'CApcCache',
        ),
        'user' => array(
            // enable cookie-based authentication
            'allowAutoLogin' => true,
        ),
        // uncomment the following to enable URLs in path-format

        'urlManager' => array(
            'urlFormat' => 'path',
            'rules' => array(
                'sitemap.xml' => 'site/sitemapxml',
                '<controller:\w+>/<id:\d+>' => '<controller>/view',
                '<controller:\w+>/<action:\w+>/<id:\d+>' => '<controller>/<action>',
                '<controller:\w+>/<action:\w+>' => '<controller>/<action>',
            ),
        ),
        //'db'=>array(
        //'connectionString' => 'sqlite:'.dirname(__FILE__).'/../data/testdrive.db',
        //),
        // uncomment the following to use a MySQL database

        'db' => array(
            'connectionString' => 'mysql:host=localhost;dbname=cbm',
            'schemaCachingDuration' => 3600,
            'emulatePrepare' => true,
            'emulatePrepare' => true,
            'username' => 'harveybc',
            'password' => '0ptimus',
            'charset' => 'utf8',
            'initSQLs' => array('set global max_allowed_packet = 1000000000',),
            //TODO: REMOVER EN PRODUCCION:
            'enableProfiling' => true,
            'enableParamLogging' => true,
        ),
        'cache' => array(
            'class' => 'CMemCache',
            'servers' => array(
                array(
                    'host' => 'localhost',
                    'port' => 11211,
                ),
            ),
        ),
        'errorHandler' => array(
            // use 'site/error' action to display errors
            'errorAction' => 'site/error',
        ),
        'log' => array(
            'class' => 'CLogRouter',
            'routes' => array(
                array(
                    'class' => 'CFileLogRoute',
                    'levels' => 'error, warning',
                ),
                // TODO: COMENTAR EN PRODUCCION 
                /*
                  array(
                  'class' => 'CWebLogRoute',
                  ),
                 *
                 
                array(
                    'class' => 'ext.yii-debug-toolbar.YiiDebugToolbarRoute',
                    'ipFilters' => array('127.0.0.1', '10.11.0.10','10.11.0.1'),
                ),
                /*
                array(
                    'class' => 'ext.db_profiler.DbProfileLogRoute',
                    'countLimit' => 1, // How many times the same query should be executed to be considered inefficient
                    'slowQueryMin' => 0.01, // Minimum time for the query to be slow
                ),
                 */
                 
            ),
        ),
    ),
    // application-level parameters that can be accessed
    // using Yii::app()->params['paramName']
    'params' => array(
        // this is used in contact page
        'adminEmail' => 'webmaster@example.com',
    ),
        // controlador por defecto
        //'defaultController' => 'site', 
);