
<style>
    #block_l5_scoreboard2,#block_l5_scoreboard,#block_l5_operations,#block_l5_menus,.operations
    ,.bcenter{
        display: none;
    }
        .hBreadcrumbs,.hLinkInicio
    {
        display:none;
    }
    
    
#block_l4_l3_menusHorizontal,#block_l5_bizquierdo,#nav,#mainMbMenu,.operaciones,#h_sidebar
    {
        display: none;
        width:0px;
    }
    .bcentro{
        width:100%;
        float:none;
    }
    #block_l4_headers{
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
    
    </style>

<?php
$this->layout = "responsiveLayout";
//$this->layout = "column1";
$this->pageTitle = Yii::app()->name . ' - Login';
$this->breadcrumbs = array(
    'Login',
);
?>
<style>
    fieldset {   
        -moz-border-radius:7px;  
        border-radius: 7px;  
        -webkit-border-radius: 7px; 
        border: 1px solid;
        border-color: #ff0000;
    }
</style>
<?php $this->pageTitle='Bienvenido a CBM - Cervecería del Valle'; ?>

<hr style="border-color:#ac8b3a;color:#ac8b3a;background-color:#ac8b3a;height: 2px;"/>
<p>Software de Administración de Base de Datos de CBM</p>


<style>
    .row, input{
        text-align: left !important;
        
    }
</style>

<div style="display:inline-block; width:100%;">
    <!--[if lt IE 9]>
        <div class="forms100cb" style="background-color:#FED;border: 2px solid #FA0;"><b style="color:#F00;">ATENCIÓN: </b><b>Se ha detectado que su navegador Web no soporta HTML5/CSS3 y puede ser inseguro.</b><br/>Haga click <a href="/chrome/GoogleChromeframeStandaloneEnterprise.msi">aquí</a> para instalar "<a href="/chrome/GoogleChromeframeStandaloneEnterprise.msi">Google Chrome Frame</a>" que <b>no requiere permisos de administrador</b> o actualice su navegador. Puede usar la aplicación pero experimentara bajo rendimiento.<br/></div>
        <![endif]-->

    <p style="text-align:left;">Por favor ingrese sus credenciales:</p>
    <div class="form" style="display:inline-block; float: left;width:40%;"">

        <?php
        $form = $this->beginWidget('CActiveForm', array(
            'id' => 'login-form',
            'enableClientValidation' => true,
            'clientOptions' => array(
                'validateOnSubmit' => true,
            ),
                ));
        ?>


        <div class="row">
            <?php echo $form->labelEx($model, 'Username', array('style' => 'width:80px;')); ?>
            <?php echo $form->textField($model, 'Username', array('style' => 'width:140px;')); ?>
            <?php echo $form->error($model, 'Username'); ?>
        </div>

        <div class="row">
            <?php echo $form->labelEx($model, 'password', array('style' => 'width:80px;')); ?>
            <?php echo $form->passwordField($model, 'password', array('style' => 'width:140px;')); ?>
            <?php echo $form->error($model, 'password'); ?>

        </div>

        <div class="row rememberMe">
            <?php echo $form->checkBox($model, 'rememberMe', array('style' => 'width:auto;')); ?>
            Recordarme
            <?php echo $form->error($model, 'rememberMe'); ?>
        </div>

        <div class="row buttons" style="width:auto; text-align:left;" >
            <?php echo CHtml::submitButton('Ingresar', array('style' => 'width:auto;')); ?>
        </div>

        <?php $this->endWidget(); ?>
    </div><!-- form -->
    <!-- sbmovil -->
    <div style="display:inline-block;width:50%;min-width: 225px;">
        <fieldset style="text-align:center;border-color:#DDDDDD;display:inline-block;width:100%; padding:10px;">
            <legend>Acceso a scoreboard móvil</legend>
            <div style="text-align:center;display:inline-block;width:25%;min-width: 100px;margin-bottom: 10px;">
                <fieldset style="text-align:center;border-color:#ac8b3a;margin-bottom: 10px;">
                    <b>BlackBerry</b>
                </fieldset>
                <?php
                $this->widget('application.extensions.qrcode.QRCodeGenerator', array(
                    'data' => 'http://cbm.ingeni-us.com:62280/scoreboardBB.php',
                    'errorCorrectionLevel' => 'M',
                    'matrixPointSize' => 3,));
                ?>
            </div>
            <div style="text-align:center;display:inline-block;width:25%;margin-left:9%;margin-right: 9%;min-width: 100px;margin-bottom: 10px;">                    
                <fieldset style="text-align:center;border-color:#ac8b3a;margin-bottom: 10px;">
                    <b>iPhone</b>
                </fieldset>
                <img src="/images/qrIP.png">
            </div>                                                
            <div style="text-align:center;display:inline-block;width:25%;min-width: 100px;margin-bottom: 10px;">                    
                <fieldset style="text-align:center;border-color:#ac8b3a;margin-bottom: 10px;">
                    <b> Android</b>
                </fieldset>
                <img src="/images/qrA.png">
            </div>
        </fieldset>  
    </div>
</div>



