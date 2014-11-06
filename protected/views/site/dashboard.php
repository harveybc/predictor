<style>
    #block_l4_headers,#block_l5_logos,#breadcrumbs_bar
    {
        display: none;
    }
    
    .operaciones,#h_sidebar,#block_l5_bizquierdo
    {
        display: none;
    }
    #block_super_headers{
        display:inline-block;
    }
    .logoIzq
    {  
        min-width: 100px;
        width:30%;
        float:left;
    }
    .logoDer
    {
        min-width: 100px;
        width:30%;
        float:right;
        display: inline-block;
    }
    #block_l5_bcentro{
        widht:100% !important;
    }
</style>


    <?php
//$this->layout = 'column1';
// $this->layout = '//layouts/responsiveLayout';
?>


<?php $this->pageTitle = Yii::app()->name; ?>



<?php
$this->pageTitle = Yii::app()->name . ' - Inicio';
$this->breadcrumbs = array(
    'Inicio',
);

//echo '<p>'.$browser.'</p>';
/* Detecta la versión de IE y selecciona el layout más adecuado. */
?>

    <!--[if lt IE 9]>
        <div class="forms100cb" style="background-color:#FED;border: 2px solid #FA0;"><b style="color:#F00;">ATENCIÓN: </b><b>Se ha detectado que su navegador Web no soporta HTML5/CSS3 y puede ser inseguro.</b><br/>Haga click <a href="/chrome/GoogleChromeframeStandaloneEnterprise.msi">aquí</a> para instalar "<a href="/chrome/GoogleChromeframeStandaloneEnterprise.msi">Google Chrome Frame</a>" que <b>no requiere permisos de administrador</b> o actualice su navegador. Puede usar la aplicación pero experimentara bajo rendimiento.<br/></div>
        <![endif]-->

<?php $this->pageTitle = 'Cervecería del Valle - Aplicaciones en Línea' ?>
<div id="h_Arbol" class="h_Arbol" style="text-align:center;">
    <p>Por favor, haga click sobre la aplicación que desee utilizar.</p>
    <div style="width:80%;text-align: center;vertical-align: middle;padding: 0;margin:0;">
        <div style="width:47%;padding: 0;margin:0;float:left;display: inline-block;">
                                <?php
                $this->widget('zii.widgets.jui.CJuiButton', array(
    'buttonType'=>'link',
    'name'=>'btnCBM',
    'caption'=>'CBM (Mantenimiento Basado en Condición)',
    'options'=>array('icons'=>'js:{secondary:"ui-icon-signal-diag"}'),
    'url'=>'http://cbm:81/index.php/site/inicio',
        // 'cssFile'=>'custom-theme/jquery-ui-1.8.21.custom.css',
        // 'themeUrl'=>'/themes',
    'htmlOptions'=>array('style'=>'width:100%;height:60px;margin-bottom:10px;vertical-align:middle;'),        
                    
));
?>  
                                                        <?php
                $this->widget('zii.widgets.jui.CJuiButton', array(
    'buttonType'=>'link',
    'name'=>'btnFORMACION',
    'caption'=>'FORMACIÓN (Administración de Recursos de Entrenamiento y Desarrollo)',
    'options'=>array('icons'=>'js:{secondary:"ui-icon-signal-diag"}'),
    'url'=>'http://cbm:84',
        // 'cssFile'=>'custom-theme/jquery-ui-1.8.21.custom.css',
        // 'themeUrl'=>'/themes',
    'htmlOptions'=>array('style'=>'width:100%;height:60px;margin-bottom:10px;vertical-align:middle;'),        
                    
));
?>  

        </div>
        <div style="width:47%;padding: 0;margin:0;float:right;display: inline-block;">
                                                        <?php
                $this->widget('zii.widgets.jui.CJuiButton', array(
    'buttonType'=>'link',
    'name'=>'btnSGDOC',
    'caption'=>'SGDOC (Sistema de Gestión Documental)',
    'options'=>array('icons'=>'js:{secondary:"ui-icon-signal-diag"}'),
    'url'=>'http://cbm:82',
        // 'cssFile'=>'custom-theme/jquery-ui-1.8.21.custom.css',
        // 'themeUrl'=>'/themes',
    'htmlOptions'=>array('style'=>'width:100%;height:60px;margin-bottom:10px;vertical-align:middle;'),        
                    
));
?>  
                                                        <?php
                $this->widget('zii.widgets.jui.CJuiButton', array(
    'buttonType'=>'link',
    'name'=>'btnMANTENIMIENTO',
    'caption'=>'AUTOMATIZACIÓN (Monitoreo de Máquinas, Eficiencias, Inventarios, Redes y Seguridad)',
    'options'=>array('icons'=>'js:{secondary:"ui-icon-signal-diag"}'),
    'url'=>'http://cbm:85',
        // 'cssFile'=>'custom-theme/jquery-ui-1.8.21.custom.css',
        // 'themeUrl'=>'/themes',
    'htmlOptions'=>array('style'=>'width:100%;height:60px;margin-bottom:10px;vertical-align:middle;'),        
                    
));
?>  
        </div>
    </div>
</div>

