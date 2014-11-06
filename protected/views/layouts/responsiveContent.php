<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN" "http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd">
<html xmlns="http://www.w3.org/1999/xhtml" xml:lang="en" lang="en" debug="true" >
    <head>
        <meta http-equiv="Content-Type" content="text/html; charset=utf-8" />
        <meta name="language" content="en" />
        <meta http-equiv="X-UA-Compatible" content="chrome=1"></meta>
        <!-- blueprint CSS framework -->
        <link rel="stylesheet" type="text/css" href="<?php echo Yii::app()->request->baseUrl; ?>/css/responsiveScreen.css" media="screen, projection" />
        <link rel="stylesheet" type="text/css" href="<?php echo Yii::app()->request->baseUrl; ?>/css/print.css" media="print" />
        <!--[if lt IE 8]>
        <link rel="stylesheet" type="text/css" href="<?php echo Yii::app()->request->baseUrl; ?>/css/ie.css" media="screen, projection" />
        <![endif]-->
        <!--[if lt IE 9]>
        <![endif]-->
        <link rel="stylesheet" type="text/css" href="<?php echo Yii::app()->request->baseUrl; ?>/css/responsiveMain.css" />
        <link rel="stylesheet" type="text/css" href="<?php echo Yii::app()->request->baseUrl; ?>/css/form.css" />
        <link rel="stylesheet" type="text/css" href="<?php echo Yii::app()->request->baseUrl; ?>/css/responsiveMediaQueries.css" />
        <link rel="stylesheet" type="text/css" href="<?php echo Yii::app()->request->baseUrl; ?>/css/responsiveMediaQueries_1.css" />
        <link rel="stylesheet" type="text/css" href="<?php echo Yii::app()->request->baseUrl; ?>/css/responsiveMediaQueries_2.css" />
        <link rel="stylesheet" type="text/css" href="<?php echo Yii::app()->request->baseUrl; ?>/css/responsiveMediaQueries_3.css" />
        <link rel="stylesheet" type="text/css" href="<?php echo Yii::app()->request->baseUrl; ?>/css/responsiveMediaQueries_4.css" />
        <link rel="stylesheet" type="text/css" href="<?php echo Yii::app()->request->baseUrl; ?>/css/responsiveMediaQueries_5.css" />
        <link rel="stylesheet" type="text/css" href="<?php echo Yii::app()->request->baseUrl; ?>/css/responsiveTree.css" />
        <!--[if lt IE 9]>
    <style>
    body {
        margin: auto;
        min-width: 850px;
        max-width: 1000px;
        _width: 900px;
    }
    #main {
        width: 55%;
    }
    #complementary {
        width: 25%;
        *margin-right: -1px; /* rounding error */
    }
    #aside {
        width: 20%;
    }
    #contentinfo {
        clear:both;
    }
    .hBreadcrumbs,.hBreadcrumbs div,.hBreadcrumbs div a,.hBreadcrumbs div span,.hLinkInicio a{
        font-size:10px !important;
    }
    .hPageTitle h2,    h1 
    {
        font-size:19px;
    }
    #layout5, .h_Arbol,
    input,html,body, div, span, object, iframe,  h3, h4, h5, h6, p, blockquote, pre, a, abbr, acronym, address, code, del, dfn, em, img, q, dl, dt, dd, ol, ul, li, fieldset, form, label, legend, table, caption, tbody, tfoot, thead, tr, th, td, article, aside, dialog, figure, footer, header, hgroup, nav, section 
    {
        font-size:12px !important;
    }
    input{
        height:14px;
    }
    .buttons input{
        height:22px;
    }

    .layout_3
    {
        text-align:center !important;
        width:100%;
    }
    #nav li ul li 
    {
        width:100%;
    } 
    #nav li ul li a
    {
        width:100%;
    } 
    #nav li ul 
    {
        margin-left: -33px;margin-top:24px;
    } 
    #nav li
    {
        vertical-align:middle;
    }
    #nav li a 
    
    {
        vertical-align: middle;
        width:available;
    } 
    #nav li a span 
    { margin-top:1px;
        vertical-align: middle;
    }
    #block_l4_headers
    {
        width:100%;
    }
    .logoIzq
    {  
        maxn-width: 257px !important;
        width:25%;
        float:left;
    }
    .logoDer
    {
        max-width: 257px !important;
        width:25%;
        float:right;
        display: inline-block;
    }
    .logoIzq img, 
    .logoDer img 
    {
        width:100%;
    }
    .menuHorizontal
    {
        margin-bottom:0px;

    }

    .layout_3 menusHorizontal,
    .operationsVertical,
    .logos,
    .scoreboardVertical
    {
        width:100%;
        display: inline-block;
        border:#c7cccf 1px solid;
        box-shadow: 2px 2px 2px rgba(0, 0, 0, 0.256875);
        -moz-box-shadow:2px 2px 2px rgba(0, 0, 0, 0.256875);
        -webkit-box-shadow:2px 2px 2px rgba(0, 0, 0, 0.256875);
        -moz-border-radius:8px;
        -webkit-border-radius: 8px;
        border-radius:8px;
        padding: 8px;  
        margin-bottom:10px;
    }
    .scoreboardVertical
    {
        width:100%;
        background-color:#fcf9ff;
    }
    .layout_3 menus
    {
       
        opacity:0.9;
    }
    .operationsH,.scoreboardH
    {
        display: inline-block;
        text-align:center;
        list-style-type: none;
        
    }
    .operationsH ul 
    {
         width:100%; margin: 0px; padding: 0px;
        display: inline-block;
            border:#811e24 0px solid;        
    box-shadow: 0px 0px 0px rgba(0, 0, 0, 0.256875);
-moz-box-shadow:0px 0px 0px rgba(0, 0, 0, 0.256875);
-webkit-box-shadow:0px 0px 0px rgba(0, 0, 0, 0.256875);
    }
    
    .operationsH,
        #nav li,#nav,#nav ul, #nav-bar
    {
        padding-bottom:2px;margin-bottom:2px;
        display: inline-block !important;
    }
    .operationsH li
    {
        display: inline-block;
    }
    #nav li a span
    { 
        font-size: 12px;
    }
    
    #nav a
    { padding: 0px 7px 0px 7px;
        
        display: inline-block !important;
    }
    .operationsH li
    {
            vertical-align: middle;
    }  
    .operationsH li a
    {
        padding-left:5px !important;padding-right:5px !important;
        display: inline-block; text-align: center;height:19px;font-weight:bold; margin: 0px; font: 12px Arial,sans-serif; 
    color: #ffffff !important; margin: 2px 0px; text-shadow: 1px 1px 1px rgba(0,0,0,0.3) !important; text-decoration: none; 
    vertical-align: middle;
    background-color: #473fdd;
    border:#195b98 1px solid;        
    box-shadow: 2px 2px 2px rgba(0, 0, 0, 0.256875);
-moz-box-shadow:2px 2px 2px rgba(0, 0, 0, 0.256875);
-webkit-box-shadow:2px 2px 2px rgba(0, 0, 0, 0.256875);
        -moz-border-radius:10px;
        -webkit-border-radius: 10px;
        border-radius:10px;
background-image: linear-gradient(bottom, rgb(77,101,138) 25%, rgb(126,173,224) 72%);
background-image: -o-linear-gradient(bottom, rgb(77,101,138) 25%, rgb(126,173,224) 72%);
background-image: -moz-linear-gradient(bottom, rgb(77,101,138) 25%, rgb(126,173,224) 72%);
background-image: -webkit-linear-gradient(bottom, rgb(77,101,138) 25%, rgb(126,173,224) 72%);
background-image: -ms-linear-gradient(bottom, rgb(77,101,138) 25%, rgb(126,173,224) 72%);

background-image: -webkit-gradient(
        linear,
        left bottom,
        left top,
        color-stop(0.25, rgb(77,101,138)),
        color-stop(0.72, rgb(126,173,224))
);
opacity: 0.95;
    }

    .operationsH li a:hover
    {
        text-align:center;
        background: #9fc9f5;
    }
    .headers,
    .content_2c_1
    {
        margin-top: 0px;
        display: inline-block;
    }
    .graphs
    {
        margin-top: 15px;
        display: inline-block;
    }
    .graphs50
    {
        margin-top: 15px;
        display: inline-block;
    }
    

    .bcentro
    {   
        width: 98%;
        min-width: 450px;
        border:#c7cccf 0px solid;
        box-shadow: 0px 0px 0px rgba(0, 0, 0, 0.256875);
        -moz-box-shadow:0px 0px 0px rgba(0, 0, 0, 0.256875);
        -webkit-box-shadow:0px 0px 0px rgba(0, 0, 0, 0.256875);
        -moz-border-radius:8px;
        -webkit-border-radius: 8px;
        border-radius:8px;
        padding: 0px 8px 8px 8px;
        


    }
        .bizquierdo,

        .layout_1,
    .layout_2,
    #block_l4_bcenter,
    .layout_5,
    .bcenter
    {
        display: none;
    }
        /************** ALL LEVELS  *************/
.operationsH li { position:relative; text-align:center; margin-left:2px;margin-right:2px;}
.operationsH li.over { z-index:99; }
.operationsH li.active { z-index:100; }
.operationsH a,
.operationsH a:hover { display:block; text-decoration:none; }
.operationsH span { display:block; }
.operationsH a { line-height:1.3em; }


/************ 1ST LEVEL  ***************/
.operationsH li { float:left; background:url(nav1_sep.gif) no-repeat 100% 0;  }
.operationsH li.active { margin-left:-1px; background:url(nav1_active.gif) no-repeat; color:#fff; font-weight:bold;  }
.operationsH li.active em { display:block; position:absolute; top:0; right:-1px; width:3px; height:27px; background:url(nav1_active.gif) no-repeat 100% 0; }
.operationsH a { float:left; padding:0 14px; color:#fff; line-height:27px; }
.operationsH li.over a { color:#d6e2e5; }


/************ 1ST LEVEL RESET ************/
.operationsH ul li,
.operationsH ul li.active { list-style-image:none;list-style-position:outside;list-style-type:none;margin:0;padding:0; float:none; height:auto; background:none; margin:0; }
.operationsH ul a,
.operationsH ul a:hover { float:none; padding:0; line-height:1.3em; }
.operationsH ul li.over a,
.operationsH ul li.over a:hover,
.operationsH ul a,
.operationsH li.active li { font-weight:normal; }
        
        
        
#nav-bar { border-top:1px solid #2d444f; border-bottom:1px solid #2d444f; background:url(nav1_bg.gif) repeat-x 0 100% #961C1F; padding:0 30px;  }
#nav {  margin:0;list-style-image:none;list-style-position:outside;list-style-type:none;margin:0;padding:0;}

/************** ALL LEVELS  *************/
#nav li { position:relative; text-align:center; }
#nav li.over { z-index:99; }
#nav li.active { z-index:100; }
#nav a,
#nav a:hover { display:block; text-decoration:none; }
#nav span { display:block; }
#nav a { line-height:1.3em; }


/************ 1ST LEVEL  ***************/
#nav li { float:left; background:url(nav1_sep.gif) no-repeat 100% 0;  }
#nav li.active { margin-left:-1px; background:url(nav1_active.gif) no-repeat; color:#fff; font-weight:bold;  }
#nav li.active em { display:block; position:absolute; top:0; right:-1px; width:3px; height:27px; background:url(nav1_active.gif) no-repeat 100% 0; }
#nav a { float:left; padding:0 14px; color:#fff; line-height:27px; }
#nav li.over a { color:#d6e2e5; }


/************ 1ST LEVEL RESET ************/
#nav ul li,
#nav ul li.active { list-style-image:none;list-style-position:outside;list-style-type:none;margin:0;padding:0; float:none; height:auto; background:none; margin:0; }
#nav ul a,
#nav ul a:hover { float:none; padding:0; line-height:1.3em; }
#nav ul li.over a,
#nav ul li.over a:hover,
#nav ul a,
#nav li.active li { font-weight:normal; }


/************ 2ND LEVEL ************/
#nav ul { list-style-image:none;list-style-position:outside;list-style-type:none;margin:0;padding:0 0 3px 0; position:absolute; width:189px; top:27px; left:-10000px; border-top:1px solid #2d444f; }
#nav ul ul  { list-style-image:none;list-style-position:outside;list-style-type:none;margin:0;padding:2px 0 0 0; border-top:0; background:url(nav3_bg.png) 0 0 no-repeat; left:100px; top:13px; }

/* Show menu */
#nav li.over ul { left:-1px;}
#nav li.over ul ul { left:-1px; }
#nav li.over ul li.over ul { left:100px; }

#nav ul li { background:url(nav2_li_bg.png) repeat-y; padding:0 2px; }
#nav ul li a { background:#F7EBA6; }
#nav ul li a:hover { background:#ffffff; }
#nav li.over ul a,
#nav ul li.active a,
#nav ul li a,
#nav ul li a:hover { color:#2f2f2f; }
#nav ul span,
#nav ul li.last li span { padding:5px 15px; background:url(nav2_link_bg.gif) repeat-x 0 100%; }
#nav ul li.last span,
#nav ul li.last li.last span { background:none; }
#nav ul li.last { background:url(nav2_last_li_bg.png) no-repeat 0 100%; padding-bottom:3px; }

#nav ul li.parent a,
#nav ul li.parent li.parent a { background-image:url(nav2_parent_arrow.gif); background-position:100% 100%; background-repeat:no-repeat; }
#nav ul li.parent li a,
#nav ul li.parent li.parent li a { background-image:none; }

/************ 3RD+ LEVEL ************/
/* Cursors */
#nav li.parent a,
#nav li.parent li.parent a,
#nav li.parent li.parent li.parent a { cursor:default; }

#nav li.parent li a,
#nav li.parent li.parent li a,
#nav li.parent li.parent li.parent li a { cursor:pointer; }

/* Show menu */
#nav ul ul ul { left:-10000px; list-style-image:none;list-style-position:outside;list-style-type:none;margin:0;padding:0; }
#nav li.over ul li.over ul ul { left:-10000px;}
#nav li.over ul li.over ul li.over ul { left:100px; }

#nav-bar:after, #nav-container:after { content:"."; display:block; clear:both; font-size:0; line-height:0; height:0; overflow:hidden; }

        </style>
<![endif]-->
        <?php //Yii::app()->clientScript->registerScriptFile(Yii::app()->baseUrl.'/css/modernizr.custom.17293.js'); ?>

        <title><?php echo CHtml::encode($this->pageTitle); ?></title>
    </head>
    <body>
        <div class="container container_3d" id="page">

            <?php echo $content; ?>

            <div id="footer" align="right" style="text-align: right;" class="footer"><span>
                    Copyright &copy; <?php echo date('Y'); ?> by Ingeni-us Soluciones de Ingeniería S.A.S.<br/>
                    Todos los Derechos Reservados.<br/>
                    <?php // echo Yii::powered(); ?>
                </span>
            </div><!-- footer -->
        </div><!-- page -->
    </body>
</html>

