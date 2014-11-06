<style type="text/css">



    .secuencias{

        width: 290px;
        
    }
</style>

<p class="note">Campos con<span class="required">*</span> son necesarios.</p>

<?php echo $form->errorSummary($model); ?>


<div styler="width:620px!important;height:216px;margin-left:0px;margin-bottom:5px !important;border-color:#961C1F;padding-top:17px;padding-right:5px;padding-left:14px; ">

<div>
    <div>
        <div>
            <label for="Usuarios">Usuario que realizó esta anotación:</label><?php
$this->widget('application.components.Relation', array(
    'model' => $model,
    'relation' => 'usuario0',
    'fields' => 'Username',
    'allowEmpty' => false,
    'style' => 'dropdownlist',
    'htmlOptions' => array(
    'class'=>'secuencias',)
        )
);
?>
        </div>
        <div>
            <label for="Documentos">La anotación pertenece a este documento:</label><?php
            $this->widget('application.components.Relation', array(
                'model' => $model,
                'relation' => 'documento0',
                'fields' => 'descripcion',
                'allowEmpty' => false,
                'style' => 'dropdownlist',
                'htmlOptions' => array(
                'class'=>'secuencias',)
                    )
            );
?>
        </div>
    </div>

</div>
<div>
    <div>
        <div>
            <div class="row">
                <?php echo $form->labelEx($model, 'descripcion'); ?>
<?php echo $form->textArea($model, 'descripcion', array('size' => 60, 'maxlength' => 256, 'style' => 'width:590px')); ?>
<?php echo $form->error($model, 'descripcion'); ?>
            </div>
        </div>
        </div>
<div>
        <div> 
            <div class="row">
                <?php echo $form->labelEx($model, 'eliminado'); ?>
<?php echo $form->checkBox($model, 'eliminado'); ?>
<?php echo $form->error($model, 'eliminado'); ?>
            </div>


        </div>
</div>
</div>
</div>







