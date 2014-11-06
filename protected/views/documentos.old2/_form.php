<style type="text/css">



    .secuencias{

        width: 200px;
        
        
    }



</style>

<p class="note">Campos con<span class="required">*</span> son necesarios.</p>

<?php echo $form->errorSummary($model); ?>
<div styler="width:607px!important;margin-left:0px;margin-bottom:5px !important;border-color:#961C1F;padding-top:10px;padding-right:5px;padding-left:14px;">
<div>
    <div>
        <div>           
            <div class="row">
                <?php echo $form->labelEx($model, 'descripcion'); ?>
                <?php echo $form->textArea($model, 'descripcion', array('size' => 60, 'maxlength' => 128, 'style' => 'width:580px')); ?>
                <?php echo $form->error($model, 'descripcion'); ?>
            </div>
        </div>
    </div>
</div>

<div styler="vertical-align:middle">
    <div>
        <div>
<div class="row">
    <?php echo $form->labelEx($model, 'permitirAdiciones'); ?>
    <?php echo $form->checkBox($model, 'permitirAdiciones'); ?>
    <?php echo $form->error($model, 'permitirAdiciones'); ?>
</div>
        </div>
        <div>
<div class="row">
    <?php echo $form->labelEx($model, 'permitirAnotaciones'); ?>
    <?php echo $form->checkBox($model, 'permitirAnotaciones'); ?>
    <?php echo $form->error($model, 'permitirAnotaciones'); ?>
</div>
        </div>
        <div>
<div class="row">
    <?php echo $form->labelEx($model, 'autorizarOtros'); ?>
    <?php echo $form->checkBox($model, 'autorizarOtros'); ?>
    <?php echo $form->error($model, 'autorizarOtros'); ?>
</div>
        </div>
        <div>
<div class="row">
    <?php echo $form->labelEx($model, 'requiereAutorizacion'); ?>
    <?php echo $form->checkBox($model, 'requiereAutorizacion'); ?>
    <?php echo $form->error($model, 'requiereAutorizacion'); ?>
</div>
        </div>
        <div>

        </div>
    </div>
</div>


<div>
    <div>
        <div>
            <div class="row">
    <?php echo $form->labelEx($model, 'conservacionInicio'); ?>
    <?php
    $this->widget('zii.widgets.jui.CJuiDatePicker', array(
        'model' => '$model',
        'name' => 'Documentos[conservacionInicio]',
        //'language'=>'de',
        'language' => 'es',
        'value' => $model->conservacionInicio,
        'htmlOptions' => array('size' => 10, 'style' => 'width:80px !important'),
        'themeUrl' => '/themes',
                    'theme' => 'calendarioCbm',
        'options' => array(
            
            'dateFormat' => 'yy-mm-dd',
                        'showButtonPanel' => true,
                        "yearRange" => '1995:2070',
                        'changeYear' => true,
                        'buttonImage' => '/images/calendar.png',
                        'showOn' => "both",
                        'buttonText' => "Seleccione la fecha",
                        'buttonImageOnly' => true,
        ),
            )
    );
    ;
    ?>
<?php echo $form->error($model, 'conservacionInicio'); ?>
</div>
        </div>
            <div>
                <div class="row">
    <?php echo $form->labelEx($model, 'conservacionFin'); ?>
    <?php
    $this->widget('zii.widgets.jui.CJuiDatePicker', array(
        'model' => '$model',
        'name' => 'Documentos[conservacionFin]',
        //'language'=>'de',
        'value' => $model->conservacionFin,
        'htmlOptions' => array('size' => 10, 'style' => 'width:80px !important'),
        'themeUrl' => '/themes',
                    'theme' => 'calendarioCbm',
        'options' => array(
            'dateFormat' => 'yy-mm-dd',
                        'showButtonPanel' => true,
                        "yearRange" => '1995:2070',
                        'changeYear' => true,
                        'buttonImage' => '/images/calendar.png',
                        'showOn' => "both",
                        'buttonText' => "Seleccione la fecha",
                        'buttonImageOnly' => true,
        ),
            )
    );
    ;
    ?>
<?php echo $form->error($model, 'conservacionFin'); ?>
</div>
            </div>
            <div>
                <div class="row">
    <?php echo $form->labelEx($model, 'conservacionPermanente'); ?>
<?php echo $form->checkBox($model, 'conservacionPermanente'); ?>
<?php echo $form->error($model, 'conservacionPermanente'); ?>
</div>
            </div>
        
       
        </div>
    </div>


<div styler="padding-left:50px;">
    <div>
        <div>
            <label for="Secuencias">Pertenece a Secuencia</label><?php
$this->widget('application.components.Relation', array(
    'model' => $model,
    'relation' => 'secuencia0',
    'fields' => 'descripcion',
    'allowEmpty' => TRUE,
    'style' => 'dropdownlist',
    'htmlOptions' => array(
    'class'=>'secuencias',)
        )
);
?>

            </div>
            <div>
                <label for="OrdenSecuencias">Orden dentro de secuencia</label><?php
$this->widget('application.components.Relation', array(
    'model' => $model,
    'relation' => 'ordenSecuencia0',
    'fields' => 'posicion',
    'allowEmpty' => true,
    'style' => 'dropdownlist',
    'htmlOptions' => array(
    'class'=>'secuencias',)
        )
);
?>
            </div>
        </div>
    </div>



</div>













			