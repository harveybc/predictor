<?php $this->beginContent('//layouts/responsiveContent'); ?>
<?php

function dibujaMenu($obj) {
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



    $obj->widget('application.extensions.mbmenu.MbMenu', array(
        'cssFile' => '/themes/mbMenu/mbmenuVertical.css',
        'items' => array(
            array('label' => 'Inicio', 'url' => array('/site/inicio'), 'visible' => !Yii::app()->user->isGuest,),
            array('label' => 'Motores', 'url' => array('/site/page', 'view' => 'motoresElectricos'), 'visible' => !Yii::app()->user->isGuest,
                'items' => array(
                    array('label' => 'Aislamiento', 'url' => array('/aislamiento_tierra/admin')),
                    array('label' => 'Vibraciones y Temperatura', 'url' => array('/vibraciones/admin')),
                    array('label' => 'Termografía Motores', 'url' => array('/termomotores/admin')),
                    array('label' => 'Lubricación', 'url' => array('/aceitesnivel1/admin'), 'visible' => !Yii::app()->user->isGuest),
                    //array('label' => 'Vibraciones 2do Nivel', 'url' => array('http://cbm.5790/')),
                    array('label' => 'Resumen de Resultados', 'url' => array('/site/page', 'view' => 'resumen')),
                ),
            ),
            array('label' => 'Ultrasonido', 'url' => array('/reportes/admin'), 'visible' => !Yii::app()->user->isGuest),
            array('label' => 'Termografía', 'url' => array('/termotablero/admin')),
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
            array('label' => 'Aplicaciones', 'url' => 'http://cbm:81/index.php/site/dashboard', 'visible' => !Yii::app()->user->isGuest),
            array('label' => 'CondmasterWEB', 'url' => 'http://cbm:5790', 'visible' => !Yii::app()->user->isGuest),
            //array('label' => 'Login', 'url' => array('/site/login'), 'visible' => Yii::app()->user->isGuest),
            array('label' => 'Salir(' . Yii::app()->user->name . ')', 'url' => array('/site/logout'), 'visible' => !Yii::app()->user->isGuest),
        ),
    ));
}
?>
<div class="container" id="hMainContainer">
    <!-- Para layout_5 : 1367 + -->
    <div id="layout_5">
        
            <div id="block_super_headers" class="forms100c">

                <div class="logoDer" id="block_l5_logoDer">
                    <img src="/images/logoDer.png"  class="logoDer"/>
                </div>
                <div class="logoIzq" id="block_l5_logoIzq" >
                    <img src="/images/logoIzq.png" class="logoIzq"/>
                </div>
                <div class="hPageTitle forms100c">
                    <h2>Bienvenido a CBM - Cervecería del Valle</h2>
                </div>

                <hr style="border-color:#ac8b3a;color:#ac8b3a;background-color:#ac8b3a;height: 2px;"/>

            </div>
        
        <div id="block_l5_bizquierdo" class="bizquierdo layout_5">
            <div id="block_l5_logos" class="logos layout_5">
                <div class="grid-block logoDer layout_5" id="block_l5_logoDer">
                    <img src="/images/logoDer.png"  class="logoDer"/>
                </div>
                <div class="grid-block layout_5" id="block_l5_logoIzq" >
                    <img src="/images/logoIzq.png" class="logoIzq"/>
                </div>

            </div>

            <div class="grid-block menus  layout_5" id="block_l5_menus" style="text-align:center;">
                <hr/>
                Menú
                <div id="mainMbMenu">

                    <?php dibujaMenu($this); ?>       
                </div>
            </div>

            <div class="grid-block operationsVertical  layout_5" id="block_l5_operations">
                <hr/>
                Operaciones
                <?php
                $this->widget('zii.widgets.CMenu', array(
                    'items' => $this->menu,
                    'htmlOptions' => array('class' => 'operationsH'),
                ));
                ?>
            </div>
<!--
            <div class="grid-block scoreboardVertical  layout_5" id="block_l5_scoreboard">
                <hr/>
                
                Eficiencia <select id="sb_graph_range"style="font-size: 11px;width:85%;" onchange="actualizaSB()">
  <option value="10">Ultimos 10 min</option>
  <option value="60">Ultima hora</option>
  <option value="360">Ultimas 6 horas</option>
    <option value="720" selected="selected">Ultimas 12 horas</option>
      <option value="1440">Ultimo dia</option>
      <option value="10080">Ultima semana</option>
      <option value="43200">Ultimo mes</option>
</select>
                <div style="line-height: 14px;height:118px;">
                     <div id="sbL1">Cargando...</div>
                </div>
            </div>
-->
        </div>
        <div id="block_l5_bcentro" style="display:inline-block" class="bcentro">
            <!-- Para layout_4 : 1024 a 1366 -->
            <div id="block_l4_headers" class="headers layout_4 logos2">
                <div class="grid-block logoDer" id="block_l4_logoDer">
                    <img src="/images/logoDer.png"  class="logoDer"/>
                </div>
                <div class="grid-block logoIzq" id="block_l4_logoIzq">
                    <img src="/images/logoIzq.png"  class="logoIzq"/>
                </div>
            </div>

            <div style="margin-top:4px;" class="grid-block menus menuHorizontal" id="block_l4_l3_menusHorizontal">
                <div id="mainMbMenu">
                    <?php dibujaMenu($this); ?>       
                </div>
            </div>

            <div class="grid-block operations layout_3" id="block_l3_operations">
                <?php
                $this->widget('zii.widgets.CMenu', array(
                    'items' => $this->menu,
                    'htmlOptions' => array('class' => 'operationsH'),
                ));
                ?>
            </div>

            <div class="grid-block menu_icons layout_2" id="block_l2_menu_icons">

            </div>
            <div class="grid-block operations_icons layout_2" id="block_l2_operations_icons">

            </div>
            <div id="content_2c" style="text-align:center;" class="content_2c grid-block">
                <div class="forms100c" id="breadcrumbs_bar">
                    <!-- Línea decorativa superior -->
                    <div style="padding: 0px;margin:0px 0px 0px 0px;display: block;" class="lineaSuperior">
                        <?php if (!Yii::app()->user->isGuest) echo '<hr style="border-color:#ac8b3a;color:#ac8b3a;background-color:#ac8b3a;height: 2px;margin: 0px 0px 0px 0px;"/>'; ?>
                    </div>
                    <!-- Breadcrumbs-->
                    <div style="width:20%;padding: 0px;margin:4px 0px 0px 0px;display: inline-block;float: left;text-align:left;" class="hBreadcrumbs">
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
                    <!-- Título de la PÁgina -->
                    <div style="width:57%;padding: 0px;margin:4px 0px 0px 0px;display: inline-block;text-align:center;" class="hPageTitle">                 
                        <?php echo '<h2>' . $this->PageTitle . '</h2>'; ?>
                    </div>
                    <!-- Link superior a Inicio -->
                    <div style="width:  20%;padding: 0px;margin:4px 0px 0px 0px;display: inline-block;float: right;font-size: 0.9em;text-align: right;" class="hLinkInicio">
                        <?php if (!Yii::app()->user->isGuest) echo CHtml::link(Yii::t('app', 'Ir a vista de árbol'), '/index.php/site/inicio', array('style' => 'text-align:right;')); ?>                            
                    </div>
                </div>
                <!-- CONTENIDO -->
                <div class="content_inner">
                    <!-- Print content -->
                    <?php echo $content; ?>
                </div>
            </div>
        </div>
    </div>
</div>
<script>
    //actualiza el Scoreboard al cargar la página
    //actualizaSB();
    // timer en javascript que ejecuta actualizaSB cada 9 segundos
    //setInterval ( "actualizaSB()", 8000 );
    //función que llama a acción KPI_L1/dynamicInfo con ajax y coloca el resultado en #sbL1
    function actualizaSB( )// here is the magic
    {

<?php
echo CHtml::ajax(array(
    'url' => array('/KPI_L1/dynamicInfo'),
    'data' => array('sb_graph_range'=>'js:$("#sb_graph_range").val()'),
    'type' => 'post',
    'update' => '#sbL1', //selector to update
));
?>
        return false;
    }
    
</script>  
<?php $this->endContent(); ?>

