
<?php echo $form->errorSummary($model); ?>

<div class="forms50cb">

<div styler="width: 100%;">
    
    <div>
        <div>
            <b>Area:</b>
    <?php
    if (!isset($model->Proceso))
        if ($valor!="")
            $model->Proceso=$valor;
// dibuja el dropDownList de Proceso, seleccionando los valores diferentes presentes en la tabla Estructura col. Proceso
echo CHtml::activedropDownList(
     $model,'Proceso', CHtml::listData(Estructura::model()->findAllbySql(
                        'SELECT DISTINCT Proceso FROM estructura ORDER BY Proceso ASC', array()), 'Proceso', 'Proceso'
        ), array(
    //'onfocus' => 'updateFieldArea()',
    //'onchange' => 'updateFieldArea()',
    'style' => 'width:100%;',
    'empty'=>'Seleccione el área',
    'class'=>'select',
        )
);
?>
                </div>
                
                </div>
    <div>
        <div>

	<div class="row">
		<?php echo $form->labelEx($model,'Tipo_Aceite'); ?>
<?php echo $form->textField($model,'Tipo_Aceite',array('size'=>50,'maxlength'=>50)); ?>
<?php echo $form->error($model,'Tipo_Aceite'); ?>
	</div>
            </div>
            
            </div>
        
                
                </div>


</div>





<!--    <div>
            
            <div>

	

--->