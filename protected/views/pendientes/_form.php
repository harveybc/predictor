<p class="note">Campos con<span class="required">*</span> son necesarios.</p>

<?php echo $form->errorSummary($model); ?>
<div>
    <div>
        <div>

            <div class="row">
                <?php echo $form->labelEx($model, 'revisado'); ?>
                <?php echo $form->checkBox($model, 'revisado'); ?>
                <?php echo $form->error($model, 'revisado'); ?>
            </div>
        </div>
        <div>
            <div class="row">
                <?php echo $form->labelEx($model, 'fecha_enviado'); ?>
                <?php
                $this->widget('zii.widgets.jui.CJuiDatePicker', array(
                    'model' => '$model',
                    'name' => 'Pendientes[fecha_enviado]',
                    //'language'=>'de',
                    'value' => $model->fecha_enviado,
                    'htmlOptions' => array('size'  => 10, 'style' => 'width:80px !important'),
                    'options' => array(
                        'showButtonPanel' => true,
                        'changeYear' => true,
                        'changeYear' => true,
                    ),
                        )
                );
                ;
                ?>
<?php echo $form->error($model, 'fecha_enviado'); ?>
            </div>
        </div>
        <div>
            <div class="row">
                <?php
                echo $form->labelEx($model, 'fecha_revisado');
                $today = date("Y-m-d");
                ?>
                <?php
                $this->widget('zii.widgets.jui.CJuiDatePicker', array(
                    'model' => '$model',
                    'name' => 'Pendientes[fecha_revisado]',
                    //'language'=>'de',
                    'value' => $today,
                    'htmlOptions' => array('size' => 10, 'style' => 'width:80px !important'),
                    'options' => array(
                        'showButtonPanel' => true,
                        'changeYear' => true,
                        'changeYear' => true,
                    ),
                        )
                );
                ;
                ?>
<?php echo $form->error($model, 'fecha_revisado'); ?>
            </div>
        </div>
    </div>
</div>

<!-- COMENTADO 

        <div class="row">
<?php echo $form->labelEx($model, 'ruta'); ?>
<?php echo $form->textField($model, 'ruta', array('size' => 60, 'maxlength' => 256)); ?>
<?php echo $form->error($model, 'ruta'); ?>
        </div>

        <div class="row">
<?php echo $form->labelEx($model, 'usuario'); ?>
<?php echo $form->textField($model, 'usuario'); ?>
<?php echo $form->error($model, 'usuario'); ?>
        </div>
-->

