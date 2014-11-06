<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN" "http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd">
<html xmlns="http://www.w3.org/1999/xhtml" xml:lang="en" lang="en" debug="true">
    <head>
        <meta http-equiv="Content-Type" content="text/html; charset=utf-8" />
        <meta name="language" content="en" />
        <!-- resetea todos los estilos -->
        <style>*{  margin: 0;  padding: 0;  }</style>
        <!-- blueprint CSS framework -->
        <link rel="stylesheet" type="text/css" href="<?php echo Yii::app()->request->baseUrl; ?>/css/screen.css" media="screen, projection" />
        <link rel="stylesheet" type="text/css" href="<?php echo Yii::app()->request->baseUrl; ?>/css/print.css" media="print" />
        <!--[if lt IE 8]>
        <link rel="stylesheet" type="text/css" href="<?php echo Yii::app()->request->baseUrl; ?>/css/ie.css" media="screen, projection" />
        <![endif]-->
        <link rel="stylesheet" type="text/css" href="<?php echo Yii::app()->request->baseUrl; ?>/css/main.css" />
        <link rel="stylesheet" type="text/css" href="<?php echo Yii::app()->request->baseUrl; ?>/css/form.css" />
        <link rel="stylesheet" type="text/css" href="<?php echo Yii::app()->request->baseUrl; ?>/css/column1.css" />
        <link rel="stylesheet" type="text/css" href="<?php echo Yii::app()->request->baseUrl; ?>/css/column1_tree.css" />

        <title><?php echo CHtml::encode($this->pageTitle); ?></title>
    </head>
    <body>

        <div class="container" id="page" >
            <div id="header" style='background: white url("/images/header_bavaria.png") no-repeat left top; width:100%;height:150px;'>
                <div id="whitespace" style="width:100%;height:85px;">
                </div>
                <div id="mainMbMenu" style="width:100%; z-index:2; float:left;" >
                    <?php
                    //TODO: provisional: para uso de roles de admin, ingeniero y usuario.
                    $esAdmin = 0;
                    $esIngeniero = 0;
                    if (!Yii::app()->user->isGuest) {
                        $modeloU = Usuarios::model()->findBySql('select * from usuarios where Username="' . Yii::app()->user->name . '"');
                    }
                    if (isset($modeloU)) {
                        $esAdmin = $modeloU->Es_administrador;
                        $esIngeniero = $modeloU->Es_analista;
                        if ($esAdmin)
                            $esIngeniero = 1;
                    }



                    $this->widget('application.extensions.mbmenu.MbMenu', array(
                        'cssFile' => '/themes/mbMenu_original/mbmenu.css',
                        'items' => array(
                            array('label' => 'Inicio', 'url' => array('/site/inicio'), 'visible' => !Yii::app()->user->isGuest,),
                            array('label' => 'Motores Eléctricos', 'url' => array('/site/page', 'view' => 'motoresElectricos'), 'visible' => !Yii::app()->user->isGuest,
                                'items' => array(
                                    array('label' => 'Aislamiento', 'url' => array('/aislamiento_tierra/admin')),
                                    array('label' => 'Vibraciones y Temperatura', 'url' => array('/vibraciones/admin')),
                                    array('label' => 'Termografía', 'url' => array('/termomotores/admin')),
                                    array('label' => 'Lubricación', 'url' => array('/aceitesnivel1/admin'), 'visible' => !Yii::app()->user->isGuest),
                                    //array('label' => 'Vibraciones 2do Nivel', 'url' => array('http://cbm.5790/')),
                                    array('label' => 'Resumen de Resultados', 'url' => array('/site/page', 'view' => 'resumen')),
                                ),
                            ),
                            array('label' => 'Ultrasonido', 'url' => array('/reportes/admin'), 'visible' => !Yii::app()->user->isGuest),
                            array('label' => 'Tableros', 'url' => array('/site/page', 'view' => 'termografia'), 'visible' => !Yii::app()->user->isGuest,
                                'items' => array(
                                    array('label' => 'Termografía', 'url' => array('/termotablero/admin')),
                                ),
                            ),
                            array('label' => 'Administrar', 'url' => array('/site/page', 'view' => 'administrar'), 'visible' => !Yii::app()->user->isGuest,
                                'items' => array(
                                    array('label' => 'Avisos ZI', 'url' => array('/avisosZI/admin'), 'visible' => $esIngeniero),
                                    array('label' => 'Gestión de Eventos', 'url' => array('/eventos/admin'), 'visible' => $esAdmin),
                                    array('label' => 'Gestión de Equipos', 'url' => array('/estructura/admin')),
                                    array('label' => 'Creación de Procesos', 'url' => array('/estructura/createProceso')),
                                    array('label' => 'Gestión de Motores', 'url' => array('/motores/admin')),
                                    array('label' => 'Gestión de Tableros', 'url' => array('/tableros/admin')),
                                    array('label' => 'Gestión de Lubricantes', 'url' => array('/tipo/admin'), 'visible' => $esIngeniero),
                                    array('label' => 'Usuarios', 'url' => array('/usuarios/admin'), 'visible' => $esAdmin),
                                    //array('label' => 'Tablero Virtual CBM', 'url' => array('/site/page', 'view' => 'tableroVirtualCbm')),
                                    //array('label' => 'Back Up de BD', 'url' => array('/site/page', 'view' => 'backup')),
                                    array('label' => 'Buscar por OT', 'url' => array('/site/page', 'view' => 'buscarOT')),
                                    array('label' => 'Mapa del Sitio', 'url' => array('/site/page', 'view' => 'sitemap'), 'visible' => !Yii::app()->user->isGuest),
                                ),
                            ),
                            array('label' => 'Documentación', 'url' => 'http://cbm:82', 'visible' => !Yii::app()->user->isGuest),
                            array('label' => 'CondmasterWEB', 'url' => 'http://cbm:5790', 'visible' => !Yii::app()->user->isGuest),
                            
                            //array('label' => 'Login', 'url' => array('/site/login'), 'visible' => Yii::app()->user->isGuest),
                            array('label' => 'Logout (' . Yii::app()->user->name . ')', 'url' => array('/site/logout'), 'visible' => !Yii::app()->user->isGuest),
                        ),
                    ));
                    ?>       
                </div>
                <div id="h_sidebar" style="float:left; background-color: #558; border: #113 3px solid;width:99%;" >
                    <?php
                    $this->widget('zii.widgets.CMenu', array(
                        'items' => $this->menu,
                        'htmlOptions' => array('class' => 'operaciones'),
                    ));
                    ?>
                </div>
            </div><!-- header -->

            <div  id="contenido" style="widht:100%;" >
                <div>
                    <!-- Línea decorativa superior -->
                    <div style="padding: 0px 10px;" class="lineaSuperior">
                        <?php if (!Yii::app()->user->isGuest) echo '<hr style="border-color:#ac8b3a;color:#ac8b3a;background-color:#ac8b3a;height: 2px;margin: 0px 0px 0px 0px;"/>'; ?>
                    </div>
                    <div id="areasup" style="padding: 0px 10px 0 10px;">
                        <!-- Breadcrumbs-->
                        <div style="width:400px;display:inline;text-align:left; float:left;">
                            <?php if (isset($this->breadcrumbs) && (!Yii::app()->user->isGuest)): ?>
                                <?php
                                echo '<b style="padding: 0px;margin:0px 4px 0px 0px;display: inline-block;float: left;">Ubicación:</b>';
                                $this->widget('zii.widgets.CBreadcrumbs', array(
                                    'links' => $this->breadcrumbs,
                                    'htmlOptions' => array('style' => 'display:inline-block;float:left;'),
                                ));
                                ?><!-- breadcrumbs -->
                            <?php endif ?>
                        </div>

                        <!-- Link superior a Inicio -->
                        <div style="width:  400px;display:inline;text-align: right;float:right;" >
                            <?php if (!Yii::app()->user->isGuest) echo CHtml::link(Yii::t('app', 'Ir a vista de árbol'), '/index.php/site/inicio', array('style' => 'text-align:right;')); ?>                            
                        </div>

                    </div>


                </div> 
                <!-- Título de la PÁgina -->
                <div style="width:100%;text-align:center;" >                 
                    <?php echo '<h2>' . $this->PageTitle . '</h2>'; ?>
                </div>
                
                <div>
                    <?php echo $content; ?>
                </div>


            </div>  
            <div id="footer" align="right" style="text-align: right;">
                Copyright &copy; <?php echo date('Y'); ?> by Ingeni-us Soluciones de Ingeniería S.A.S.<br/>
                Todos los Derechos Reservados.<br/>
                <?php // echo Yii::powered();  ?>

            </div><!-- footer -->
        </div><!-- page -->
    </body>
</html>

